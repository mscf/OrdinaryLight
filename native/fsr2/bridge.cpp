#include "upstream/ffx_fsr2.h"
#include "upstream/vk/ffx_fsr2_vk.h"
#include <vector>
#include <new>
struct Context { FfxFsr2Context fsr{}; std::vector<uint8_t> scratch; uint32_t width,height; };
extern "C" {
int ol_fsr2_version() { return 1; }
void* ol_fsr2_create(void* physical,void* device,uint32_t w,uint32_t h,int* error) {
    auto c=new(std::nothrow) Context(); if(!c){*error=-1;return nullptr;}
    c->width=w;c->height=h;
    c->scratch.resize(ffxFsr2GetScratchMemorySizeVK((VkPhysicalDevice)physical));
    FfxFsr2ContextDescription d{}; d.flags=FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE|FFX_FSR2_ENABLE_DEPTH_INVERTED;
    d.maxRenderSize={w,h};d.displaySize={w,h};d.device=ffxGetDeviceVK((VkDevice)device);
    *error=ffxFsr2GetInterfaceVK(&d.callbacks,c->scratch.data(),c->scratch.size(),(VkPhysicalDevice)physical,vkGetDeviceProcAddr);
    if(!*error)*error=ffxFsr2ContextCreate(&c->fsr,&d);
    if(*error){delete c;return nullptr;}return c;
}
void ol_fsr2_destroy(void* ptr) {auto c=(Context*)ptr;if(c){ffxFsr2ContextDestroy(&c->fsr);delete c;}}
void ol_fsr2_jitter(int frame,int width,int output,float* x,float* y) {
    ffxFsr2GetJitterOffset(x,y,frame,ffxFsr2GetJitterPhaseCount(width,output));
}
int ol_fsr2_dispatch(void* ptr,void* command,const uint64_t* images,const uint64_t* views,uint32_t w,uint32_t h,float jx,float jy,float fov,float dt,int reset) {
    auto c=(Context*)ptr;FfxFsr2DispatchDescription d{};
    d.commandList=ffxGetCommandListVK((VkCommandBuffer)command);
    auto res=[&](int i,VkFormat fmt){return ffxGetTextureResourceVK(&c->fsr,(VkImage)images[i],(VkImageView)views[i],i==4?c->width:w,i==4?c->height:h,fmt,L"OrdinaryLight",FFX_RESOURCE_STATE_UNORDERED_ACCESS);};
    d.color=res(0,VK_FORMAT_R16G16B16A16_SFLOAT);d.depth=res(1,VK_FORMAT_R32_SFLOAT);
    d.motionVectors=res(2,VK_FORMAT_R16G16_SFLOAT);d.reactive=res(3,VK_FORMAT_R8_UNORM);
    d.output=res(4,VK_FORMAT_R16G16B16A16_SFLOAT);
    d.jitterOffset={jx,jy};d.motionVectorScale={1,1};d.renderSize={w,h};
    d.frameTimeDelta=dt;d.preExposure=1;d.reset=reset;d.cameraNear=.1f;d.cameraFar=10000;d.cameraFovAngleVertical=fov;d.viewSpaceToMetersFactor=1;
    int error=ffxFsr2ContextDispatch(&c->fsr,&d);
    // SDK leaves external inputs in compute-read layout. Restore our GENERAL contract.
    VkImageMemoryBarrier b[4]{};
    for(int i=0;i<4;i++){b[i].sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;b[i].srcAccessMask=VK_ACCESS_SHADER_READ_BIT;b[i].dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;b[i].oldLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;b[i].newLayout=VK_IMAGE_LAYOUT_GENERAL;b[i].srcQueueFamilyIndex=b[i].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;b[i].image=(VkImage)images[i];b[i].subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};}
    vkCmdPipelineBarrier((VkCommandBuffer)command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,0,nullptr,4,b);
    return error;
}
}
