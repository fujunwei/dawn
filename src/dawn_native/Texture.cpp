// Copyright 2017 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dawn_native/Texture.h"

#include <algorithm>

#include "common/Assert.h"
#include "common/Constants.h"
#include "common/Math.h"
#include "dawn_native/Device.h"
#include "dawn_native/ValidationUtils_autogen.h"

namespace dawn_native {
    namespace {
        // TODO(jiawei.shao@intel.com): implement texture view format compatibility rule
        MaybeError ValidateTextureViewFormatCompatibility(const TextureBase* texture,
                                                          const TextureViewDescriptor* descriptor) {
            if (texture->GetFormat().format != descriptor->format) {
                return DAWN_VALIDATION_ERROR(
                    "The format of texture view is not compatible to the original texture");
            }

            return {};
        }

        // TODO(jiawei.shao@intel.com): support validation on all texture view dimensions
        bool IsTextureViewDimensionCompatibleWithTextureDimension(
            dawn::TextureViewDimension textureViewDimension,
            dawn::TextureDimension textureDimension) {
            switch (textureViewDimension) {
                case dawn::TextureViewDimension::e2D:
                case dawn::TextureViewDimension::e2DArray:
                case dawn::TextureViewDimension::Cube:
                case dawn::TextureViewDimension::CubeArray:
                    return textureDimension == dawn::TextureDimension::e2D;
                default:
                    UNREACHABLE();
                    return false;
            }
        }

        // TODO(jiawei.shao@intel.com): support validation on all texture view dimensions
        bool IsArrayLayerValidForTextureViewDimension(
            dawn::TextureViewDimension textureViewDimension,
            uint32_t textureViewArrayLayer) {
            switch (textureViewDimension) {
                case dawn::TextureViewDimension::e2D:
                    return textureViewArrayLayer == 1u;
                case dawn::TextureViewDimension::e2DArray:
                    return true;
                case dawn::TextureViewDimension::Cube:
                    return textureViewArrayLayer == 6u;
                case dawn::TextureViewDimension::CubeArray:
                    return textureViewArrayLayer % 6 == 0;
                default:
                    UNREACHABLE();
                    return false;
            }
        }

        bool IsTextureSizeValidForTextureViewDimension(
            dawn::TextureViewDimension textureViewDimension,
            const Extent3D& textureSize) {
            switch (textureViewDimension) {
                case dawn::TextureViewDimension::Cube:
                case dawn::TextureViewDimension::CubeArray:
                    return textureSize.width == textureSize.height;
                case dawn::TextureViewDimension::e2D:
                case dawn::TextureViewDimension::e2DArray:
                    return true;
                default:
                    UNREACHABLE();
                    return false;
            }
        }

        // TODO(jiawei.shao@intel.com): support more sample count.
        MaybeError ValidateSampleCount(const TextureDescriptor* descriptor, const Format* format) {
            if (!IsValidSampleCount(descriptor->sampleCount)) {
                return DAWN_VALIDATION_ERROR("The sample count of the texture is not supported.");
            }

            if (descriptor->sampleCount > 1) {
                if (descriptor->mipLevelCount > 1) {
                    return DAWN_VALIDATION_ERROR(
                        "The mipmap level count of a multisampled texture must be 1.");
                }

                // Multisampled 2D array texture is not supported because on Metal it requires the
                // version of macOS be greater than 10.14.
                if (descriptor->arrayLayerCount > 1) {
                    return DAWN_VALIDATION_ERROR("Multisampled 2D array texture is not supported.");
                }

                if (format->isCompressed) {
                    return DAWN_VALIDATION_ERROR(
                        "The sample counts of the textures in BC formats must be 1.");
                }
            }

            return {};
        }

        MaybeError ValidateTextureViewDimensionCompatibility(
            const TextureBase* texture,
            const TextureViewDescriptor* descriptor) {
            if (!IsArrayLayerValidForTextureViewDimension(descriptor->dimension,
                                                          descriptor->arrayLayerCount)) {
                return DAWN_VALIDATION_ERROR(
                    "The dimension of the texture view is not compatible with the layer count");
            }

            if (!IsTextureViewDimensionCompatibleWithTextureDimension(descriptor->dimension,
                                                                      texture->GetDimension())) {
                return DAWN_VALIDATION_ERROR(
                    "The dimension of the texture view is not compatible with the dimension of the"
                    "original texture");
            }

            if (!IsTextureSizeValidForTextureViewDimension(descriptor->dimension,
                                                           texture->GetSize())) {
                return DAWN_VALIDATION_ERROR(
                    "The dimension of the texture view is not compatible with the size of the"
                    "original texture");
            }

            return {};
        }

        MaybeError ValidateTextureSize(const TextureDescriptor* descriptor, const Format* format) {
            ASSERT(descriptor->size.width != 0 && descriptor->size.height != 0);

            if (Log2(std::max(descriptor->size.width, descriptor->size.height)) + 1 <
                descriptor->mipLevelCount) {
                return DAWN_VALIDATION_ERROR("Texture has too many mip levels");
            }

            if (format->isCompressed && (descriptor->size.width % format->blockWidth != 0 ||
                                         descriptor->size.height % format->blockHeight != 0)) {
                return DAWN_VALIDATION_ERROR(
                    "The size of the texture is incompatible with the texture format");
            }

            return {};
        }

        MaybeError ValidateTextureUsage(const TextureDescriptor* descriptor, const Format* format) {
            DAWN_TRY(dawn_native::ValidateTextureUsage(descriptor->usage));

            constexpr dawn::TextureUsage kValidCompressedUsages = dawn::TextureUsage::Sampled |
                                                                  dawn::TextureUsage::CopySrc |
                                                                  dawn::TextureUsage::CopyDst;
            if (format->isCompressed && (descriptor->usage & (~kValidCompressedUsages))) {
                return DAWN_VALIDATION_ERROR(
                    "Compressed texture format is incompatible with the texture usage");
            }

            if (!format->isRenderable &&
                (descriptor->usage & dawn::TextureUsage::OutputAttachment)) {
                return DAWN_VALIDATION_ERROR(
                    "Non-renderable format used with OutputAttachment usage");
            }

            if (descriptor->usage & dawn::TextureUsage::Storage) {
                return DAWN_VALIDATION_ERROR("storage textures aren't supported (yet)");
            }

            return {};
        }

    }  // anonymous namespace

    MaybeError ValidateTextureDescriptor(const DeviceBase* device,
                                         const TextureDescriptor* descriptor) {
        if (descriptor == nullptr) {
            return DAWN_VALIDATION_ERROR("Texture descriptor is nullptr");
        }
        if (descriptor->nextInChain != nullptr) {
            return DAWN_VALIDATION_ERROR("nextInChain must be nullptr");
        }

        const Format* format;
        DAWN_TRY_ASSIGN(format, device->GetInternalFormat(descriptor->format));

        DAWN_TRY(ValidateTextureUsage(descriptor, format));
        DAWN_TRY(ValidateTextureDimension(descriptor->dimension));
        DAWN_TRY(ValidateSampleCount(descriptor, format));

        // TODO(jiawei.shao@intel.com): check stuff based on the dimension
        if (descriptor->size.width == 0 || descriptor->size.height == 0 ||
            descriptor->size.depth == 0 || descriptor->arrayLayerCount == 0 ||
            descriptor->mipLevelCount == 0) {
            return DAWN_VALIDATION_ERROR("Cannot create an empty texture");
        }

        if (descriptor->dimension != dawn::TextureDimension::e2D) {
            return DAWN_VALIDATION_ERROR("Texture dimension must be 2D (for now)");
        }

        DAWN_TRY(ValidateTextureSize(descriptor, format));

        return {};
    }

    MaybeError ValidateTextureViewDescriptor(const TextureBase* texture,
                                             const TextureViewDescriptor* descriptor) {
        if (descriptor->nextInChain != nullptr) {
            return DAWN_VALIDATION_ERROR("nextInChain must be nullptr");
        }

        // Parent texture should have been already validated.
        ASSERT(texture);
        ASSERT(!texture->IsError());
        if (texture->GetTextureState() == TextureBase::TextureState::Destroyed) {
            return DAWN_VALIDATION_ERROR("Destroyed texture used to create texture view");
        }

        DAWN_TRY(ValidateTextureViewDimension(descriptor->dimension));
        if (descriptor->dimension == dawn::TextureViewDimension::e1D ||
            descriptor->dimension == dawn::TextureViewDimension::e3D) {
            return DAWN_VALIDATION_ERROR("Texture view dimension must be 2D compatible.");
        }

        DAWN_TRY(ValidateTextureFormat(descriptor->format));

        // TODO(jiawei.shao@intel.com): check stuff based on resource limits
        if (descriptor->arrayLayerCount == 0 || descriptor->mipLevelCount == 0) {
            return DAWN_VALIDATION_ERROR("Cannot create an empty texture view");
        }

        if (uint64_t(descriptor->baseArrayLayer) + uint64_t(descriptor->arrayLayerCount) >
            uint64_t(texture->GetArrayLayers())) {
            return DAWN_VALIDATION_ERROR("Texture view array-layer out of range");
        }

        if (uint64_t(descriptor->baseMipLevel) + uint64_t(descriptor->mipLevelCount) >
            uint64_t(texture->GetNumMipLevels())) {
            return DAWN_VALIDATION_ERROR("Texture view mip-level out of range");
        }

        DAWN_TRY(ValidateTextureViewFormatCompatibility(texture, descriptor));
        DAWN_TRY(ValidateTextureViewDimensionCompatibility(texture, descriptor));

        return {};
    }

    TextureViewDescriptor GetTextureViewDescriptorWithDefaults(
        const TextureBase* texture,
        const TextureViewDescriptor* descriptor) {
        ASSERT(texture);

        TextureViewDescriptor desc = {};
        if (descriptor) {
            desc = *descriptor;
        }

        // The default value for the view dimension depends on the texture's dimension with a
        // special case for 2DArray being chosen automatically if arrayLayerCount is unspecified.
        if (desc.dimension == dawn::TextureViewDimension::Undefined) {
            switch (texture->GetDimension()) {
                case dawn::TextureDimension::e1D:
                    desc.dimension = dawn::TextureViewDimension::e1D;
                    break;

                case dawn::TextureDimension::e2D:
                    if (texture->GetArrayLayers() > 1u && desc.arrayLayerCount == 0) {
                        desc.dimension = dawn::TextureViewDimension::e2DArray;
                    } else {
                        desc.dimension = dawn::TextureViewDimension::e2D;
                    }
                    break;

                case dawn::TextureDimension::e3D:
                    desc.dimension = dawn::TextureViewDimension::e3D;
                    break;

                default:
                    UNREACHABLE();
            }
        }

        if (desc.format == dawn::TextureFormat::Undefined) {
            desc.format = texture->GetFormat().format;
        }
        if (desc.arrayLayerCount == 0) {
            desc.arrayLayerCount = texture->GetArrayLayers() - desc.baseArrayLayer;
        }
        if (desc.mipLevelCount == 0) {
            desc.mipLevelCount = texture->GetNumMipLevels() - desc.baseMipLevel;
        }
        return desc;
    }

    bool IsValidSampleCount(uint32_t sampleCount) {
        switch (sampleCount) {
            case 1:
            case 4:
                return true;

            default:
                return false;
        }
    }

    // TextureBase

    TextureBase::TextureBase(DeviceBase* device,
                             const TextureDescriptor* descriptor,
                             TextureState state)
        : ObjectBase(device),
          mDimension(descriptor->dimension),
          mFormat(device->GetValidInternalFormat(descriptor->format)),
          mSize(descriptor->size),
          mArrayLayerCount(descriptor->arrayLayerCount),
          mMipLevelCount(descriptor->mipLevelCount),
          mSampleCount(descriptor->sampleCount),
          mUsage(descriptor->usage),
          mState(state) {
        uint32_t subresourceCount =
            GetSubresourceIndex(descriptor->mipLevelCount, descriptor->arrayLayerCount);
        mIsSubresourceContentInitializedAtIndex = std::vector<bool>(subresourceCount, false);
    }

    static Format kUnusedFormat;

    TextureBase::TextureBase(DeviceBase* device, ObjectBase::ErrorTag tag)
        : ObjectBase(device, tag), mFormat(kUnusedFormat) {
    }

    // static
    TextureBase* TextureBase::MakeError(DeviceBase* device) {
        return new TextureBase(device, ObjectBase::kError);
    }

    dawn::TextureDimension TextureBase::GetDimension() const {
        ASSERT(!IsError());
        return mDimension;
    }

    // TODO(jiawei.shao@intel.com): return more information about texture format
    const Format& TextureBase::GetFormat() const {
        ASSERT(!IsError());
        return mFormat;
    }
    const Extent3D& TextureBase::GetSize() const {
        ASSERT(!IsError());
        return mSize;
    }
    uint32_t TextureBase::GetArrayLayers() const {
        ASSERT(!IsError());
        return mArrayLayerCount;
    }
    uint32_t TextureBase::GetNumMipLevels() const {
        ASSERT(!IsError());
        return mMipLevelCount;
    }
    uint32_t TextureBase::GetSampleCount() const {
        ASSERT(!IsError());
        return mSampleCount;
    }
    dawn::TextureUsage TextureBase::GetUsage() const {
        ASSERT(!IsError());
        return mUsage;
    }

    TextureBase::TextureState TextureBase::GetTextureState() const {
        ASSERT(!IsError());
        return mState;
    }

    uint32_t TextureBase::GetSubresourceIndex(uint32_t mipLevel, uint32_t arraySlice) const {
        ASSERT(arraySlice <= kMaxTexture2DArrayLayers);
        ASSERT(mipLevel <= kMaxTexture2DMipLevels);
        static_assert(kMaxTexture2DMipLevels <=
                          std::numeric_limits<uint32_t>::max() / kMaxTexture2DArrayLayers,
                      "texture size overflows uint32_t");
        return GetNumMipLevels() * arraySlice + mipLevel;
    }

    bool TextureBase::IsSubresourceContentInitialized(uint32_t baseMipLevel,
                                                      uint32_t levelCount,
                                                      uint32_t baseArrayLayer,
                                                      uint32_t layerCount) const {
        ASSERT(!IsError());
        for (uint32_t mipLevel = baseMipLevel; mipLevel < baseMipLevel + levelCount; ++mipLevel) {
            for (uint32_t arrayLayer = baseArrayLayer; arrayLayer < baseArrayLayer + layerCount;
                 ++arrayLayer) {
                uint32_t subresourceIndex = GetSubresourceIndex(mipLevel, arrayLayer);
                ASSERT(subresourceIndex < mIsSubresourceContentInitializedAtIndex.size());
                if (!mIsSubresourceContentInitializedAtIndex[subresourceIndex]) {
                    return false;
                }
            }
        }
        return true;
    }

    void TextureBase::SetIsSubresourceContentInitialized(uint32_t baseMipLevel,
                                                         uint32_t levelCount,
                                                         uint32_t baseArrayLayer,
                                                         uint32_t layerCount) {
        ASSERT(!IsError());
        for (uint32_t mipLevel = baseMipLevel; mipLevel < baseMipLevel + levelCount; ++mipLevel) {
            for (uint32_t arrayLayer = baseArrayLayer; arrayLayer < baseArrayLayer + layerCount;
                 ++arrayLayer) {
                uint32_t subresourceIndex = GetSubresourceIndex(mipLevel, arrayLayer);
                ASSERT(subresourceIndex < mIsSubresourceContentInitializedAtIndex.size());
                mIsSubresourceContentInitializedAtIndex[subresourceIndex] = true;
            }
        }
    }

    MaybeError TextureBase::ValidateCanUseInSubmitNow() const {
        ASSERT(!IsError());
        if (mState == TextureState::Destroyed) {
            return DAWN_VALIDATION_ERROR("Destroyed texture used in a submit");
        }
        return {};
    }

    bool TextureBase::IsMultisampledTexture() const {
        ASSERT(!IsError());
        return mSampleCount > 1;
    }

    Extent3D TextureBase::GetMipLevelVirtualSize(uint32_t level) const {
        Extent3D extent;
        extent.width = std::max(mSize.width >> level, 1u);
        extent.height = std::max(mSize.height >> level, 1u);
        extent.depth = std::max(mSize.depth >> level, 1u);
        return extent;
    }

    Extent3D TextureBase::GetMipLevelPhysicalSize(uint32_t level) const {
        Extent3D extent = GetMipLevelVirtualSize(level);

        // Compressed Textures will have paddings if their width or height is not a multiple of
        // 4 at non-zero mipmap levels.
        if (mFormat.isCompressed) {
            // TODO(jiawei.shao@intel.com): check if there are any overflows.
            uint32_t blockWidth = mFormat.blockWidth;
            uint32_t blockHeight = mFormat.blockHeight;
            extent.width = (extent.width + blockWidth - 1) / blockWidth * blockWidth;
            extent.height = (extent.height + blockHeight - 1) / blockHeight * blockHeight;
        }

        return extent;
    }

    TextureViewBase* TextureBase::CreateView(const TextureViewDescriptor* descriptor) {
        return GetDevice()->CreateTextureView(this, descriptor);
    }

    void TextureBase::Destroy() {
        if (GetDevice()->ConsumedError(ValidateDestroy())) {
            return;
        }
        ASSERT(!IsError());
        DestroyInternal();
    }

    void TextureBase::DestroyImpl() {
    }

    void TextureBase::DestroyInternal() {
        if (mState == TextureState::OwnedInternal) {
            DestroyImpl();
        }
        mState = TextureState::Destroyed;
    }

    MaybeError TextureBase::ValidateDestroy() const {
        DAWN_TRY(GetDevice()->ValidateObject(this));
        return {};
    }

    // TextureViewBase

    TextureViewBase::TextureViewBase(TextureBase* texture, const TextureViewDescriptor* descriptor)
        : ObjectBase(texture->GetDevice()),
          mTexture(texture),
          mFormat(GetDevice()->GetValidInternalFormat(descriptor->format)),
          mBaseMipLevel(descriptor->baseMipLevel),
          mMipLevelCount(descriptor->mipLevelCount),
          mBaseArrayLayer(descriptor->baseArrayLayer),
          mArrayLayerCount(descriptor->arrayLayerCount) {
    }

    TextureViewBase::TextureViewBase(DeviceBase* device, ObjectBase::ErrorTag tag)
        : ObjectBase(device, tag), mFormat(kUnusedFormat) {
    }

    // static
    TextureViewBase* TextureViewBase::MakeError(DeviceBase* device) {
        return new TextureViewBase(device, ObjectBase::kError);
    }

    const TextureBase* TextureViewBase::GetTexture() const {
        ASSERT(!IsError());
        return mTexture.Get();
    }

    TextureBase* TextureViewBase::GetTexture() {
        ASSERT(!IsError());
        return mTexture.Get();
    }

    const Format& TextureViewBase::GetFormat() const {
        ASSERT(!IsError());
        return mFormat;
    }

    uint32_t TextureViewBase::GetBaseMipLevel() const {
        ASSERT(!IsError());
        return mBaseMipLevel;
    }

    uint32_t TextureViewBase::GetLevelCount() const {
        ASSERT(!IsError());
        return mMipLevelCount;
    }

    uint32_t TextureViewBase::GetBaseArrayLayer() const {
        ASSERT(!IsError());
        return mBaseArrayLayer;
    }

    uint32_t TextureViewBase::GetLayerCount() const {
        ASSERT(!IsError());
        return mArrayLayerCount;
    }
}  // namespace dawn_native
