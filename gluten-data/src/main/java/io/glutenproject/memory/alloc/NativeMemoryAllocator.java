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
package io.glutenproject.memory.alloc;

import io.glutenproject.memory.NativeMemoryJniWrapper;

/**
 * This along with {@link NativeMemoryAllocators}, as built-in toolkit for managing native memory
 * allocations.
 */
public class NativeMemoryAllocator {
  enum Type {
    DEFAULT,
  }

  private final long nativeInstanceId;

  public NativeMemoryAllocator(long nativeInstanceId) {
    this.nativeInstanceId = nativeInstanceId;
  }

  public static NativeMemoryAllocator create(Type type) {
    return new NativeMemoryAllocator(NativeMemoryJniWrapper.getAllocator(type.name()));
  }

  public long getNativeInstanceId() {
    return this.nativeInstanceId;
  }

  public void close() {
    NativeMemoryJniWrapper.releaseAllocator(this.nativeInstanceId);
  }
}
