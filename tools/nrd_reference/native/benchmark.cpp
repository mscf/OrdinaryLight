#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <NRI.h>
#include <Extensions/NRIDeviceCreation.h>
#include <Extensions/NRIHelper.h>
#include <Extensions/NRIRayTracing.h>
#include <Extensions/NRIWrapperVK.h>
#include <NRD.h>
#include <NRDIntegration.h>
#include <NRDIntegration.hpp>
#include <vulkan/vulkan.h>

namespace {

void check(nri::Result value, const char* operation) {
    if (value != nri::Result::SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed");
}

void check(nrd::Result value, const char* operation) {
    if (value != nrd::Result::SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed");
}

struct Options {
    uint16_t width = 1280;
    uint16_t height = 720;
    uint32_t warmup = 8;
    uint32_t iterations = 32;
    std::string input;
    std::string output;
};

Options parse(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc)
            throw std::runtime_error("missing value for " + key);
        const std::string argument = argv[++i];
        if (key == "--input") result.input = argument;
        else if (key == "--output") result.output = argument;
        else if (key == "--width") result.width = static_cast<uint16_t>(std::stoul(argument));
        else if (key == "--height") result.height = static_cast<uint16_t>(std::stoul(argument));
        else if (key == "--warmup") result.warmup = static_cast<uint32_t>(std::stoul(argument));
        else if (key == "--iterations") result.iterations = static_cast<uint32_t>(std::stoul(argument));
        else throw std::runtime_error("unknown option " + key);
    }
    if (result.input.empty() != result.output.empty())
        throw std::runtime_error("--input and --output must be provided together");
    if (!result.width || !result.height || !result.iterations)
        throw std::runtime_error("width, height, and iterations must be positive");
    return result;
}

template <typename T>
void readExact(std::istream& stream, T* destination, size_t count = 1) {
    stream.read(reinterpret_cast<char*>(destination), std::streamsize(sizeof(T) * count));
    if (!stream)
        throw std::runtime_error("truncated NRD capture");
}

struct CaptureFrame {
    uint64_t frameIndex = 0;
    uint8_t cameraCut = 0;
    float jitter[4] = {};
    float worldToClip[16] = {};
    float worldToClipPrev[16] = {};
    std::vector<uint16_t> motion;
    std::vector<uint16_t> normalRoughness;
    std::vector<float> viewZ;
    std::vector<uint16_t> diffuse;
    std::vector<uint16_t> specular;
};

std::vector<CaptureFrame> readCapture(Options& options) {
    std::ifstream stream(options.input, std::ios::binary);
    if (!stream)
        throw std::runtime_error("could not open NRD capture " + options.input);
    char magic[8];
    readExact(stream, magic, 8);
    if (std::memcmp(magic, "OLNRDIN1", 8) != 0)
        throw std::runtime_error("invalid NRD capture magic");
    uint32_t metadata[4];
    readExact(stream, metadata, 4);
    if (metadata[0] != 1 || !metadata[1] || !metadata[2] || !metadata[3] ||
        metadata[1] > UINT16_MAX || metadata[2] > UINT16_MAX)
        throw std::runtime_error("unsupported NRD capture metadata");
    options.width = static_cast<uint16_t>(metadata[1]);
    options.height = static_cast<uint16_t>(metadata[2]);
    const size_t pixels = size_t(options.width) * options.height;
    std::vector<CaptureFrame> frames(metadata[3]);
    for (CaptureFrame& frame : frames) {
        readExact(stream, &frame.frameIndex);
        readExact(stream, &frame.cameraCut);
        char padding[3]; readExact(stream, padding, 3);
        readExact(stream, frame.jitter, 4);
        readExact(stream, frame.worldToClip, 16);
        readExact(stream, frame.worldToClipPrev, 16);
        frame.motion.resize(pixels * 2); readExact(stream, frame.motion.data(), frame.motion.size());
        frame.normalRoughness.resize(pixels * 4); readExact(stream, frame.normalRoughness.data(), frame.normalRoughness.size());
        frame.viewZ.resize(pixels); readExact(stream, frame.viewZ.data(), frame.viewZ.size());
        frame.diffuse.resize(pixels * 4); readExact(stream, frame.diffuse.data(), frame.diffuse.size());
        frame.specular.resize(pixels * 4); readExact(stream, frame.specular.data(), frame.specular.size());
    }
    return frames;
}

void identity(float* matrix) {
    std::fill(matrix, matrix + 16, 0.0f);
    matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1.0f;
}

} // namespace

int main(int argc, char** argv) try {
    Options options = parse(argc, argv);
    std::vector<CaptureFrame> captureFrames;
    if (!options.input.empty())
        captureFrames = readCapture(options);
    const auto wallStart = std::chrono::steady_clock::now();

    // NRI's automatic device creation enables a broad feature/extension set
    // that is unnecessary for NRD and fails on some Linux NVIDIA driver
    // combinations. Create the smallest Vulkan 1.3 compute-capable device and
    // wrap it through NRI's public Vulkan interop API instead.
    VkApplicationInfo applicationInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    applicationInfo.pApplicationName = "ordinarylight-nrd";
    applicationInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instanceInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &applicationInfo;
    VkInstance vkInstance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instanceInfo, nullptr, &vkInstance) != VK_SUCCESS)
        throw std::runtime_error("vkCreateInstance failed");

    uint32_t physicalDeviceNum = 0;
    vkEnumeratePhysicalDevices(vkInstance, &physicalDeviceNum, nullptr);
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceNum);
    vkEnumeratePhysicalDevices(vkInstance, &physicalDeviceNum, physicalDevices.data());
    VkPhysicalDevice vkPhysicalDevice = VK_NULL_HANDLE;
    for (VkPhysicalDevice candidate : physicalDevices) {
        VkPhysicalDeviceProperties properties = {};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.vendorID == 0x10de && properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            vkPhysicalDevice = candidate;
            break;
        }
    }
    if (!vkPhysicalDevice)
        throw std::runtime_error("no NVIDIA discrete Vulkan adapter is available");

    uint32_t queueFamilyNum = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice, &queueFamilyNum, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyNum);
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice, &queueFamilyNum, queueFamilies.data());
    uint32_t queueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyNum; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queueFamilyIndex = i;
            break;
        }
    }
    if (queueFamilyIndex == UINT32_MAX)
        throw std::runtime_error("NVIDIA Vulkan adapter has no graphics queue");

    VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features features11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features features12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features features13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features.pNext = &features11;
    features11.pNext = &features12;
    features12.pNext = &features13;
    vkGetPhysicalDeviceFeatures2(vkPhysicalDevice, &features);

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    VkDeviceCreateInfo deviceInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &features;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    VkDevice vkDevice = VK_NULL_HANDLE;
    const VkResult createResult = vkCreateDevice(vkPhysicalDevice, &deviceInfo, nullptr, &vkDevice);
    if (createResult != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed with code " + std::to_string(createResult));

    nri::QueueFamilyVKDesc queueFamily = {};
    queueFamily.queueNum = 1;
    queueFamily.queueType = nri::QueueType::GRAPHICS;
    queueFamily.familyIndex = queueFamilyIndex;
    nri::DeviceCreationVKDesc wrappedDevice = {};
    wrappedDevice.vkBindingOffsets = {0, 20, 2, 3};
    wrappedDevice.vkInstance = vkInstance;
    wrappedDevice.vkDevice = vkDevice;
    wrappedDevice.vkPhysicalDevice = vkPhysicalDevice;
    wrappedDevice.queueFamilies = &queueFamily;
    wrappedDevice.queueFamilyNum = 1;
    wrappedDevice.minorVersion = 3;
    wrappedDevice.enableNRIValidation = std::getenv("ORDINARYLIGHT_NRD_VALIDATION") != nullptr;
    wrappedDevice.enableMemoryZeroInitialization = true;

    nri::Device* device = nullptr;
    check(nri::nriCreateDeviceFromVKDevice(wrappedDevice, device), "nriCreateDeviceFromVKDevice");

    nri::CoreInterface core = {};
    nri::HelperInterface helper = {};
    check(nri::nriGetInterface(*device, NRI_INTERFACE(nri::CoreInterface), &core), "CoreInterface");
    check(nri::nriGetInterface(*device, NRI_INTERFACE(nri::HelperInterface), &helper), "HelperInterface");
    const nri::DeviceDesc& deviceDesc = core.GetDeviceDesc(*device);
    if (!deviceDesc.features.timestamp || !deviceDesc.other.timestampFrequencyHz)
        throw std::runtime_error("selected Vulkan device does not expose timestamp queries");

    nri::Queue* queue = nullptr;
    check(core.GetQueue(*device, nri::QueueType::GRAPHICS, 0, queue), "GetQueue");

    const nrd::Identifier identifier = 1;
    const nrd::DenoiserDesc denoiser = {identifier, nrd::Denoiser::RELAX_DIFFUSE_SPECULAR};
    nrd::InstanceCreationDesc instanceDesc = {};
    instanceDesc.denoisers = &denoiser;
    instanceDesc.denoisersNum = 1;
    nrd::IntegrationCreationDesc integrationDesc = {};
    std::strncpy(integrationDesc.name, "ordinarylight-nrd", sizeof(integrationDesc.name) - 1);
    integrationDesc.resourceWidth = options.width;
    integrationDesc.resourceHeight = options.height;
    integrationDesc.queuedFrameNum = 1;
    integrationDesc.autoWaitForIdle = false;
    integrationDesc.enableWholeLifetimeDescriptorCaching = true;
    nrd::Integration integration;
    check(integration.Recreate(integrationDesc, instanceDesc, device), "NRD Integration::Recreate");

    struct TextureSpec { nrd::ResourceType slot; nri::Format format; };
    const TextureSpec specs[] = {
        {nrd::ResourceType::IN_MV, nri::Format::RG16_SFLOAT},
        {nrd::ResourceType::IN_NORMAL_ROUGHNESS, nri::Format::RGBA16_SFLOAT},
        {nrd::ResourceType::IN_VIEWZ, nri::Format::R32_SFLOAT},
        {nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, nri::Format::RGBA16_SFLOAT},
        {nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, nri::Format::RGBA16_SFLOAT},
        {nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, nri::Format::RGBA16_SFLOAT},
        {nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, nri::Format::RGBA16_SFLOAT},
    };
    std::vector<nri::Texture*> textures;
    textures.reserve(std::size(specs));
    for (const TextureSpec& spec : specs) {
        nri::TextureDesc desc = {};
        desc.type = nri::TextureType::TEXTURE_2D;
        desc.usage = nri::TextureUsageBits::SHADER_RESOURCE |
            nri::TextureUsageBits::SHADER_RESOURCE_STORAGE |
            nri::TextureUsageBits::HOST_TRANSFER;
        desc.format = spec.format;
        desc.width = options.width;
        desc.height = options.height;
        desc.depth = 1;
        desc.mipNum = 1;
        desc.layerNum = 1;
        desc.sampleNum = 1;
        nri::Texture* texture = nullptr;
        check(core.CreateTexture(*device, desc, texture), "CreateTexture");
        textures.push_back(texture);
    }
    nri::ResourceGroupDesc resourceGroup = {};
    resourceGroup.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroup.textures = textures.data();
    resourceGroup.textureNum = static_cast<uint32_t>(textures.size());
    std::vector<nri::Memory*> textureMemory(helper.CalculateAllocationNumber(*device, resourceGroup));
    check(helper.AllocateAndBindMemory(*device, resourceGroup, textureMemory.data()), "AllocateAndBindMemory(textures)");

    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    check(core.CreateCommandAllocator(*queue, allocator), "CreateCommandAllocator");
    check(core.CreateCommandBuffer(*allocator, commandBuffer), "CreateCommandBuffer");
    nri::QueryPoolDesc queryDesc = {nri::QueryType::TIMESTAMP, 2};
    nri::QueryPool* queryPool = nullptr;
    check(core.CreateQueryPool(*device, queryDesc, queryPool), "CreateQueryPool");
    const uint32_t querySize = core.GetQuerySize(*queryPool);
    nri::BufferDesc queryBufferDesc = {};
    queryBufferDesc.size = uint64_t(querySize) * 2;
    nri::Buffer* queryBuffer = nullptr;
    check(core.CreateCommittedBuffer(*device, nri::MemoryLocation::HOST_READBACK, 0.0f, queryBufferDesc, queryBuffer), "CreateCommittedBuffer(query)");

    nrd::RelaxSettings relax = {};
    check(integration.SetDenoiserSettings(identifier, &relax), "SetDenoiserSettings");

    const nri::AccessLayoutStage inputState = {
        nri::AccessBits::SHADER_RESOURCE,
        nri::Layout::SHADER_RESOURCE,
        nri::StageBits::COMPUTE_SHADER,
    };
    const nri::AccessLayoutStage outputState = {
        nri::AccessBits::HOST_READ,
        nri::Layout::GENERAL,
        nri::StageBits::HOST,
    };
    const size_t pixels = size_t(options.width) * options.height;
    std::vector<uint16_t> zeroHalf(pixels * 4, 0);
    auto uploadTexture = [&](size_t textureIndex, const void* data, uint32_t rowPitch,
                             const nri::AccessLayoutStage& after) {
        nri::TextureSubresourceUploadDesc subresource = {};
        subresource.slices = data;
        subresource.sliceNum = 1;
        subresource.rowPitch = rowPitch;
        subresource.slicePitch = rowPitch * options.height;
        nri::TextureUploadDesc upload = {};
        upload.subresources = &subresource;
        upload.texture = textures[textureIndex];
        upload.after = after;
        upload.planes = nri::PlaneBits::COLOR;
        check(helper.UploadData(*queue, &upload, 1, nullptr, 0), "UploadData(texture)");
    };

    // Establish truthful initial states for every externally-owned resource.
    for (size_t i = 0; i < 5; ++i) {
        const uint32_t stride = i == 0 ? 4 : (i == 2 ? 4 : 8);
        uploadTexture(i, zeroHalf.data(), options.width * stride, inputState);
    }
    uploadTexture(5, zeroHalf.data(), options.width * 8, outputState);
    uploadTexture(6, zeroHalf.data(), options.width * 8, outputState);

    if (!captureFrames.empty()) {
        std::ofstream output(options.output, std::ios::binary);
        if (!output)
            throw std::runtime_error("could not open NRD output " + options.output);
        output.write("OLNRDOU1", 8);
        const uint32_t metadata[4] = {1, options.width, options.height,
                                      static_cast<uint32_t>(captureFrames.size())};
        output.write(reinterpret_cast<const char*>(metadata), sizeof(metadata));
        std::vector<uint16_t> diffuseOutput(pixels * 4);
        std::vector<uint16_t> specularOutput(pixels * 4);
        for (size_t frameNumber = 0; frameNumber < captureFrames.size(); ++frameNumber) {
            const CaptureFrame& frame = captureFrames[frameNumber];
            uploadTexture(0, frame.motion.data(), options.width * 4, inputState);
            uploadTexture(1, frame.normalRoughness.data(), options.width * 8, inputState);
            uploadTexture(2, frame.viewZ.data(), options.width * 4, inputState);
            uploadTexture(3, frame.diffuse.data(), options.width * 8, inputState);
            uploadTexture(4, frame.specular.data(), options.width * 8, inputState);

            integration.NewFrame();
            nrd::CommonSettings common = {};
            // The canonical contract currently supplies world->clip. Treating
            // world as view space preserves the exact combined transform; a
            // future contract revision will carry the split view transform.
            std::memcpy(common.viewToClipMatrix, frame.worldToClip, sizeof(frame.worldToClip));
            std::memcpy(common.viewToClipMatrixPrev, frame.worldToClipPrev, sizeof(frame.worldToClipPrev));
            identity(common.worldToViewMatrix);
            identity(common.worldToViewMatrixPrev);
            common.motionVectorScale[0] = 1.0f / options.width;
            common.motionVectorScale[1] = 1.0f / options.height;
            common.motionVectorScale[2] = 0.0f;
            common.cameraJitter[0] = frame.jitter[0] / options.width;
            common.cameraJitter[1] = frame.jitter[1] / options.height;
            common.cameraJitterPrev[0] = frame.jitter[2] / options.width;
            common.cameraJitterPrev[1] = frame.jitter[3] / options.height;
            common.resourceSize[0] = common.resourceSizePrev[0] = common.rectSize[0] = common.rectSizePrev[0] = options.width;
            common.resourceSize[1] = common.resourceSizePrev[1] = common.rectSize[1] = common.rectSizePrev[1] = options.height;
            common.frameIndex = static_cast<uint32_t>(frame.frameIndex);
            common.accumulationMode = (frameNumber == 0 || frame.cameraCut) ?
                nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
            check(integration.SetCommonSettings(common), "SetCommonSettings(quality)");

            nrd::ResourceSnapshot snapshot;
            snapshot.restoreInitialState = true;
            for (size_t i = 0; i < std::size(specs); ++i) {
                nrd::Resource resource = {};
                resource.nri.texture = textures[i];
                resource.state = i < 5 ? inputState : outputState;
                snapshot.SetResource(specs[i].slot, resource);
            }
            core.ResetCommandAllocator(*allocator);
            check(core.BeginCommandBuffer(*commandBuffer, nullptr), "BeginCommandBuffer(quality)");
            integration.Denoise(&identifier, 1, *commandBuffer, snapshot);
            check(core.EndCommandBuffer(*commandBuffer), "EndCommandBuffer(quality)");
            const nri::CommandBuffer* buffers[] = {commandBuffer};
            nri::QueueSubmitDesc submit = {};
            submit.commandBuffers = buffers;
            submit.commandBufferNum = 1;
            check(core.QueueSubmit(*queue, submit), "QueueSubmit(quality)");
            check(core.QueueWaitIdle(queue), "QueueWaitIdle(quality)");

            nri::ReadbackTextureToHostMemoryDesc readbacks[2] = {};
            readbacks[0].srcTexture = textures[5];
            readbacks[0].dstData = diffuseOutput.data();
            readbacks[0].srcRegion = {0, 0, 0, options.width, options.height, 1, 0, 0, nri::PlaneBits::COLOR};
            readbacks[0].dstRowPitch = options.width * 8;
            readbacks[1] = readbacks[0];
            readbacks[1].srcTexture = textures[6];
            readbacks[1].dstData = specularOutput.data();
            check(core.ReadbackTextureToHostMemory(*queue, readbacks, 2), "ReadbackTextureToHostMemory");
            output.write(reinterpret_cast<const char*>(diffuseOutput.data()), std::streamsize(diffuseOutput.size() * 2));
            output.write(reinterpret_cast<const char*>(specularOutput.data()), std::streamsize(specularOutput.size() * 2));
        }
        output.close();
        core.DeviceWaitIdle(device);
        integration.Destroy();
        core.DestroyBuffer(queryBuffer);
        core.DestroyQueryPool(queryPool);
        core.DestroyCommandBuffer(commandBuffer);
        core.DestroyCommandAllocator(allocator);
        for (nri::Texture* texture : textures) core.DestroyTexture(texture);
        for (nri::Memory* memory : textureMemory) core.FreeMemory(memory);
        nri::nriDestroyDevice(device);
        vkDestroyDevice(vkDevice, nullptr);
        vkDestroyInstance(vkInstance, nullptr);
        return 0;
    }

    std::vector<double> gpuTimes;
    gpuTimes.reserve(options.iterations);
    const uint32_t total = options.warmup + options.iterations;
    for (uint32_t frame = 0; frame < total; ++frame) {
        integration.NewFrame();
        nrd::CommonSettings common = {};
        identity(common.viewToClipMatrix);
        identity(common.viewToClipMatrixPrev);
        identity(common.worldToViewMatrix);
        identity(common.worldToViewMatrixPrev);
        common.resourceSize[0] = common.resourceSizePrev[0] = common.rectSize[0] = common.rectSizePrev[0] = options.width;
        common.resourceSize[1] = common.resourceSizePrev[1] = common.rectSize[1] = common.rectSizePrev[1] = options.height;
        common.frameIndex = frame;
        common.accumulationMode = frame == 0 ? nrd::AccumulationMode::CLEAR_AND_RESTART : nrd::AccumulationMode::CONTINUE;
        check(integration.SetCommonSettings(common), "SetCommonSettings");

        nrd::ResourceSnapshot snapshot;
        snapshot.restoreInitialState = true;
        for (size_t i = 0; i < std::size(specs); ++i) {
            nrd::Resource resource = {};
            resource.nri.texture = textures[i];
            resource.state = i < 5 ? inputState : outputState;
            snapshot.SetResource(specs[i].slot, resource);
        }

        core.ResetCommandAllocator(*allocator);
        check(core.BeginCommandBuffer(*commandBuffer, nullptr), "BeginCommandBuffer");
        core.CmdResetQueries(*commandBuffer, *queryPool, 0, 2);
        core.CmdEndQuery(*commandBuffer, *queryPool, 0);
        integration.Denoise(&identifier, 1, *commandBuffer, snapshot);
        core.CmdEndQuery(*commandBuffer, *queryPool, 1);
        core.CmdCopyQueries(*commandBuffer, *queryPool, 0, 2, *queryBuffer, 0);
        check(core.EndCommandBuffer(*commandBuffer), "EndCommandBuffer");
        const nri::CommandBuffer* buffers[] = {commandBuffer};
        nri::QueueSubmitDesc submit = {};
        submit.commandBuffers = buffers;
        submit.commandBufferNum = 1;
        check(core.QueueSubmit(*queue, submit), "QueueSubmit");
        check(core.QueueWaitIdle(queue), "QueueWaitIdle");
        auto* timestamps = static_cast<const uint64_t*>(core.MapBuffer(*queryBuffer, 0, queryBufferDesc.size));
        if (!timestamps)
            throw std::runtime_error("MapBuffer(query) failed");
        const uint64_t delta = timestamps[1] - timestamps[0];
        core.UnmapBuffer(*queryBuffer);
        if (frame >= options.warmup)
            gpuTimes.push_back(double(delta) * 1000.0 / double(deviceDesc.other.timestampFrequencyHz));
    }

    std::sort(gpuTimes.begin(), gpuTimes.end());
    const double median = gpuTimes[gpuTimes.size() / 2];
    const size_t p95Index = std::min(gpuTimes.size() - 1, size_t(std::ceil(gpuTimes.size() * 0.95)) - 1);
    const double p95 = gpuTimes[p95Index];
    const double wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wallStart).count();
    std::cout << "{\n"
              << "  \"median_gpu_ms\": " << median << ",\n"
              << "  \"p95_gpu_ms\": " << p95 << ",\n"
              << "  \"wall_ms\": " << wallMs << ",\n"
              << "  \"persistent_mib\": " << integration.GetPersistentMemoryUsageInMb() << ",\n"
              << "  \"transient_mib\": " << integration.GetAliasableMemoryUsageInMb() << ",\n"
              << "  \"measured_frames\": " << gpuTimes.size() << ",\n"
              << "  \"implementation_version\": \"NRD 4.18.0 RELAX_DIFFUSE_SPECULAR\"\n"
              << "}\n";

    core.DeviceWaitIdle(device);
    integration.Destroy();
    core.DestroyBuffer(queryBuffer);
    core.DestroyQueryPool(queryPool);
    core.DestroyCommandBuffer(commandBuffer);
    core.DestroyCommandAllocator(allocator);
    for (nri::Texture* texture : textures) core.DestroyTexture(texture);
    for (nri::Memory* memory : textureMemory) core.FreeMemory(memory);
    nri::nriDestroyDevice(device);
    vkDestroyDevice(vkDevice, nullptr);
    vkDestroyInstance(vkInstance, nullptr);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ordinarylight_nrd_benchmark: " << error.what() << '\n';
    return 1;
}
