/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <velox/common/base/BitUtil.h>
#include <velox/common/memory/MemoryPool.h>
#include <velox/type/Filter.h>
#include <velox/vector/ComplexVector.h>
#include "compute/ResultIterator.h"
#include "memory/VeloxColumnarBatch.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/BaseVector.h"

namespace gluten {
class RowVectorStream {
 public:
  explicit RowVectorStream(
      facebook::velox::memory::MemoryPool* pool,
      std::shared_ptr<ResultIterator> iterator,
      const facebook::velox::RowTypePtr& outputType)
      : iterator_(iterator), outputType_(outputType), pool_(pool) {}

  bool hasNext() {
    return iterator_->hasNext();
  }

  // Convert arrow batch to rowvector and use new output columns
  facebook::velox::RowVectorPtr next() {
    const std::shared_ptr<VeloxColumnarBatch>& vb = VeloxColumnarBatch::from(pool_, iterator_->next());
    auto vp = vb->getRowVector();
    VELOX_DCHECK(vp != nullptr);
    return std::make_shared<facebook::velox::RowVector>(
        vp->pool(), outputType_, facebook::velox::BufferPtr(0), vp->size(), std::move(vp->children()));
  }

 private:
  std::shared_ptr<ResultIterator> iterator_;
  const facebook::velox::RowTypePtr outputType_;
  facebook::velox::memory::MemoryPool* pool_;
};

class ValueStreamNode : public facebook::velox::core::PlanNode {
 public:
  ValueStreamNode(
      const facebook::velox::core::PlanNodeId& id,
      const facebook::velox::RowTypePtr& outputType,
      std::shared_ptr<RowVectorStream> valueStream)
      : facebook::velox::core::PlanNode(id), outputType_(outputType), valueStream_(std::move(valueStream)) {
    VELOX_CHECK_NOT_NULL(valueStream_);
  }

  const facebook::velox::RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<facebook::velox::core::PlanNodePtr>& sources() const override {
    return kEmptySources;
  };

  const std::shared_ptr<RowVectorStream>& rowVectorStream() const {
    return valueStream_;
  }

  std::string_view name() const override {
    return "ValueStream";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("ValueStream plan node is not serializable");
  }

 private:
  void addDetails(std::stringstream& stream) const override{};

  const facebook::velox::RowTypePtr outputType_;
  std::shared_ptr<RowVectorStream> valueStream_;
  const std::vector<facebook::velox::core::PlanNodePtr> kEmptySources;
};

using namespace facebook::velox;

class ValueStream : public facebook::velox::exec::SourceOperator {
 public:
  ValueStream(
      int32_t operatorId,
      facebook::velox::exec::DriverCtx* driverCtx,
      std::shared_ptr<const ValueStreamNode> valueStreamNode)
      : facebook::velox::exec::SourceOperator(
            driverCtx,
            valueStreamNode->outputType(),
            operatorId,
            valueStreamNode->id(),
            "ValueStream") {
    valueStream_ = valueStreamNode->rowVectorStream();
  }

  facebook::velox::RowVectorPtr getOutput() override {
    using namespace facebook::velox;
    if (valueStream_->hasNext()) {
      if (!this->dynamicFilters_.empty()) {
        auto output = valueStream_->next();
        LOG(INFO) << "before df " << output->size();
        applyDynamicFilter(output);
        LOG(INFO) << "after df " << output->size();
        return output;
      } else {
        return valueStream_->next();
      }
    } else {
      finished_ = true;
      return nullptr;
    }
  }

  facebook::velox::exec::BlockingReason isBlocked(facebook::velox::ContinueFuture* /* unused */) override {
    return facebook::velox::exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

  bool canAddDynamicFilter() const override {
    return true;
  }

  void addDynamicFilter(
      facebook::velox::column_index_t index,
      const std::shared_ptr<facebook::velox::common::Filter>& filter) override {
    LOG(INFO)  << "addDynamicFilter at " << index << " with " << filter->toString();
    this->dynamicFilters_[index] = filter;
  }

 private:
  bool finished_ = false;
  std::shared_ptr<RowVectorStream> valueStream_;

  void applyDynamicFilter(RowVectorPtr vector) {
    auto originSize = vector->size();

    std::vector<VectorPtr>& children = vector->children();
    std::vector<uint64_t> passed(bits::nwords(originSize), -1);

    for (auto& [column_idx, df] : this->dynamicFilters_) {
      auto& child = children[column_idx];
      applyFilter(df, *child, passed.data());
    }

    auto size = bits::countBits(passed.data(), 0, originSize);
    if (size < originSize) {
      LOG(INFO) << "df worked, before " << originSize << " after " << size;
      auto indices = allocateIndices(size, vector->pool());
      auto* rawIndices = indices->asMutable<vector_size_t>();
      vector_size_t j = 0;
      bits::forEachSetBit(passed.data(), 0, originSize, [&](auto i) { rawIndices[j++] = i; });
      for (auto& child : children) {
        child = BaseVector::wrapInDictionary(nullptr, indices, size, std::move(child));
      }
    }
  }

  void applyFilter(const std::shared_ptr<facebook::velox::common::Filter> filter, const facebook::velox::BaseVector& vector, uint64_t* result) {
    bits::forEachSetBit(result, 0, vector.size(), [&](auto i) {
      if (!filterRow(vector, *filter, i)) {
        bits::clearBit(result, i);
      }
    });
  }

  bool filterRow(const BaseVector& vector, common::Filter& filter, vector_size_t index) {
    if (vector.isNullAt(index)) {
      return filter.testNull();
    }
    switch (vector.typeKind()) {
      case TypeKind::ARRAY:
      case TypeKind::MAP:
      case TypeKind::ROW:
        VELOX_UNSUPPORTED("Pushdown dynamic filter not support {}", vector.typeKind())
      default:
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            filterSimpleVectorRow, vector.typeKind(), vector, filter, index);
    }
  }

  template <TypeKind kKind>
  bool filterSimpleVectorRow(
      const BaseVector& vector,
      common::Filter& filter,
      vector_size_t index) {
    using T = typename TypeTraits<kKind>::NativeType;
    auto* simpleVector = vector.asUnchecked<SimpleVector<T>>();
    return common::applyFilter(filter, simpleVector->valueAt(index));
  }
};

class RowVectorStreamOperatorTranslator : public facebook::velox::exec::Operator::PlanNodeTranslator {
  std::unique_ptr<facebook::velox::exec::Operator>
  toOperator(facebook::velox::exec::DriverCtx* ctx, int32_t id, const facebook::velox::core::PlanNodePtr& node) {
    if (auto valueStreamNode = std::dynamic_pointer_cast<const ValueStreamNode>(node)) {
      return std::make_unique<ValueStream>(id, ctx, valueStreamNode);
    }
    return nullptr;
  }
};
} // namespace gluten
