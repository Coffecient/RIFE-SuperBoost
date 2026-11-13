// SPDX-License-Identifier: MIT
// Independent IPluginV3 implementation; no community exporter/plugin is used.
#include <NvInferRuntime.h>
#include <array>
#include <cstring>
#include <mutex>
#include <new>
#include "fusion.h"

namespace {
using namespace nvinfer1;
constexpr const char* names[]{"RsbStagePack", "RsbFlowWarpPack", "RsbWarpBlend"};
constexpr const char* fields[]{"features", "scale", "previous_scale", "first", "half"};
constexpr const char* space="rife_superboost";
bool dimensions(const Dims& d, int channels, int h, int w) {
    return d.nbDims==4 && d.d[0]==1 && d.d[1]==channels && d.d[2]==h && d.d[3]==w;
}
class Fusion final : public IPluginV3, public IPluginV3OneCore,
                     public IPluginV3OneBuild, public IPluginV3OneRuntime {
    int kind;
    RsbFusion params;
    std::array<PluginField,5> serialized;
    PluginFieldCollection collection{};
public:
    Fusion(int type, RsbFusion values) : kind(type), params(values) {
        // type/values: Prepared operation identity and its immutable scalar fields.
        int* data[]{&params.features,&params.scale,&params.previous_scale,&params.first,&params.half};
        for(int i=0;i<5;++i) serialized[i]=PluginField{fields[i],data[i],PluginFieldType::kINT32,1};
        collection={5,serialized.data()};
    }
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
        if(t==PluginCapabilityType::kCORE)return static_cast<IPluginV3OneCore*>(this);
        if(t==PluginCapabilityType::kBUILD)return static_cast<IPluginV3OneBuild*>(this);
        if(t==PluginCapabilityType::kRUNTIME)return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }
    IPluginV3* clone() noexcept override {return new(std::nothrow) Fusion(kind,params);}
    const char* getPluginName() const noexcept override {return names[kind];}
    const char* getPluginVersion() const noexcept override {return "1";}
    const char* getPluginNamespace() const noexcept override {return space;}
    int32_t getNbOutputs() const noexcept override {return kind==1?2:1;}
    int inputs() const {return kind==0?5:kind==1?7:4;}
    int32_t getOutputDataTypes(DataType* out,int32_t no,const DataType*,int32_t ni) const noexcept override {
        if(!out || ni!=inputs() || no!=getNbOutputs())return -1;
        out[0]=params.half?DataType::kHALF:DataType::kFLOAT;
        if(no==2)out[1]=DataType::kFLOAT;
        return 0;
    }
    int32_t getOutputShapes(const DimsExprs* in,int32_t ni,const DimsExprs*,int32_t ns,
        DimsExprs* out,int32_t no,IExprBuilder& b) noexcept override {
        if(!in || !out || ni!=inputs() || ns || no!=getNbOutputs() || in[0].nbDims!=4)return -1;
        out[0]=in[0];
        if(kind!=2) {
            out[0].d[1]=b.constant((kind==0?7:20)+2*params.features);
            auto* scale=b.constant(params.scale);
            out[0].d[2]=b.operation(DimensionOperation::kFLOOR_DIV,*in[0].d[2],*scale);
            out[0].d[3]=b.operation(DimensionOperation::kFLOOR_DIV,*in[0].d[3],*scale);
        }
        if(no==2){out[1]=in[0];out[1].d[1]=b.constant(4);}
        return 0;
    }
    bool supportsFormatCombination(int32_t pos,const DynamicPluginTensorDesc* desc,int32_t ni,int32_t no) noexcept override {
        if(!desc || ni!=inputs() || no!=getNbOutputs() || pos<0 || pos>=ni+no)return false;
        const auto& value=desc[pos].desc;
        auto neural=params.half?DataType::kHALF:DataType::kFLOAT;
        if(pos==ni)return value.type==neural && value.format==(kind!=2 && params.half?TensorFormat::kHWC8:TensorFormat::kLINEAR);
        if(pos>ni)return value.type==DataType::kFLOAT && value.format==TensorFormat::kLINEAR;
        if(value.format!=TensorFormat::kLINEAR)return false;
        if(pos<2)return (desc[0].desc.type==DataType::kFLOAT || desc[0].desc.type==DataType::kHALF) && value.type==desc[0].desc.type;
        if(kind!=2 && pos<4)return value.type==neural;
        return value.type==DataType::kFLOAT;
    }
    int32_t configurePlugin(const DynamicPluginTensorDesc*,int32_t ni,const DynamicPluginTensorDesc*,int32_t no) noexcept override {
        if(ni!=inputs() || no!=getNbOutputs() || (params.features!=4 && params.features!=16) ||
           params.scale<1 || params.scale>32 || (params.scale&(params.scale-1)) ||
           params.previous_scale<1 || params.previous_scale>32 || (params.previous_scale&(params.previous_scale-1)) ||
           (params.half!=0 && params.half!=1) || (params.first!=0 && params.first!=1))return -1;
        return 0;
    }
    int32_t onShapeChange(const PluginTensorDesc* in,int32_t ni,const PluginTensorDesc* out,int32_t no) noexcept override {
        if(!in || !out || ni!=inputs() || no!=getNbOutputs() || in[0].dims.nbDims!=4)return -1;
        int64_t hh=in[0].dims.d[2],ww=in[0].dims.d[3];
        if(hh<64 || ww<64 || hh>8192 || ww>8192)return -1;
        int h=int(hh),w=int(ww);
        if(!dimensions(in[0].dims,3,h,w) || !dimensions(in[1].dims,3,h,w))return -1;
        if(kind==2) {
            if(!dimensions(in[2].dims,4,h,w) || !dimensions(in[3].dims,5,h,w) || !dimensions(out[0].dims,3,h,w))return -1;
            return 0;
        }
        if(h%params.scale || w%params.scale || !dimensions(in[2].dims,params.features,h,w) ||
           !dimensions(in[3].dims,params.features,h,w) ||
           !dimensions(out[0].dims,(kind==0?7:20)+2*params.features,h/params.scale,w/params.scale))return -1;
        if(kind==0)return dimensions(in[4].dims,1,1,1)?0:-1;
        if(h%params.previous_scale || w%params.previous_scale ||
           !dimensions(in[4].dims,4,params.first?1:h,params.first?1:w) ||
           !dimensions(in[5].dims,13,h/params.previous_scale,w/params.previous_scale) ||
           !dimensions(in[6].dims,1,1,1) || !dimensions(out[1].dims,4,h,w))return -1;
        return 0;
    }
    int32_t enqueue(const PluginTensorDesc* in,const PluginTensorDesc*,const void* const* inputs,
        void* const* outputs,void*,cudaStream_t stream) noexcept override {
        // All shape/type/format constraints have been checked before execution.
        int bytes=in[0].type==DataType::kHALF?2:4;
        return rsb_fusion(kind,params,inputs,outputs,int(in[0].dims.d[2]),int(in[0].dims.d[3]),bytes,stream)==cudaSuccess?0:-1;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {return clone();}
    const PluginFieldCollection* getFieldsToSerialize() noexcept override {return &collection;}
    const char* getMetadataString() noexcept override {return "Official FP32 flow/geometry; sparse bilinear border warp; HWC8 neural packing; source storage explicit";}
};
class Creator final : public IPluginCreatorV3One {
    int kind;
    std::array<PluginField,5> declared;
    PluginFieldCollection collection{};
public:
    explicit Creator(int type):kind(type) {
        // type: One of the three separately serialized operator identities.
        for(int i=0;i<5;++i)declared[i]=PluginField{fields[i],nullptr,PluginFieldType::kINT32,1};
        collection={5,declared.data()};
    }
    const char* getPluginName() const noexcept override {return names[kind];}
    const char* getPluginVersion() const noexcept override {return "1";}
    const char* getPluginNamespace() const noexcept override {return space;}
    const PluginFieldCollection* getFieldNames() noexcept override {return &collection;}
    IPluginV3* createPlugin(const char*,const PluginFieldCollection* fc,TensorRTPhase) noexcept override {
        RsbFusion values;
        int* data[]{&values.features,&values.scale,&values.previous_scale,&values.first,&values.half};
        if(fc)for(int i=0;i<fc->nbFields;++i) {
            const auto& v=fc->fields[i];
            for(int k=0;k<5;++k)if(v.name && std::strcmp(v.name,fields[k])==0) {
                if(!v.data || v.length!=1 || v.type!=PluginFieldType::kINT32)return nullptr;
                *data[k]=*static_cast<const int*>(v.data);
            }
        }
        return new(std::nothrow) Fusion(kind,values);
    }
};
}
bool rsb_register_fusions() noexcept {
    static std::once_flag once;
    static bool ok=false;
    try {
        std::call_once(once,[]{
            static Creator creators[]{Creator(0),Creator(1),Creator(2)};
            auto* registry=getPluginRegistry();
            ok=registry && registry->registerCreator(creators[0],space) && registry->registerCreator(creators[1],space) && registry->registerCreator(creators[2],space);
        });
    } catch(...) {return false;}
    return ok;
}
