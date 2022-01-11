// This file is a part of Julia. License is MIT: https://julialang.org/license

#include <utility>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Debug.h>
#include <llvm/IR/DebugLoc.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/DIBuilder.h>
#include "julia.h"

#define STR(csym)           #csym
#define XSTR(csym)          STR(csym)

enum AddressSpace {
    Generic = 0,
    Tracked = 10,
    Derived = 11,
    CalleeRooted = 12,
    Loaded = 13,
    FirstSpecial = Tracked,
    LastSpecial = Loaded,
};

namespace JuliaType {

    static inline llvm::StructType* get_jlvalue_ty(llvm::LLVMContext &C) {
        return llvm::StructType::get(C);
    }

    static inline llvm::PointerType* get_pjlvalue_ty(llvm::LLVMContext &C, AddressSpace as = AddressSpace::Generic) {
        return llvm::PointerType::get(get_jlvalue_ty(C), as);
    }

    static inline llvm::PointerType* get_prjlvalue_ty(llvm::LLVMContext &C) {
        return get_pjlvalue_ty(C, AddressSpace::Tracked);
    }

    static inline llvm::PointerType* get_ppjlvalue_ty(llvm::LLVMContext &C, AddressSpace as = AddressSpace::Generic) {
        return llvm::PointerType::get(get_pjlvalue_ty(C, as), 0);
    }

    static inline llvm::PointerType* get_pprjlvalue_ty(llvm::LLVMContext &C) {
        return get_ppjlvalue_ty(C, AddressSpace::Tracked);
    }
}

struct TypeCache {
    struct {
        llvm::Type *T_jlvalue;
        llvm::Type *T_pjlvalue;
        llvm::Type *T_prjlvalue;
        llvm::Type *T_ppjlvalue;
        llvm::Type *T_pprjlvalue;
        llvm::Type *T_jlarray;
        llvm::Type *T_pjlarray;
        llvm::Type *T_pvoidfunc;
        llvm::FunctionType *T_jlfunc;
        llvm::FunctionType *T_jlfuncsparams;

        void initialize(llvm::LLVMContext &context, TypeCache &cache) {
            T_jlvalue = JuliaType::get_jlvalue_ty(context);
            T_pjlvalue = JuliaType::get_pjlvalue_ty(context);
            T_prjlvalue = JuliaType::get_prjlvalue_ty(context);
            T_ppjlvalue = JuliaType::get_ppjlvalue_ty(context);
            T_pprjlvalue = JuliaType::get_pprjlvalue_ty(context);
            std::vector<llvm::Type*> ftargs(0);
            ftargs.push_back(T_prjlvalue);  // function
            ftargs.push_back(T_pprjlvalue); // args[]
            ftargs.push_back(cache.T_int32(context));      // nargs
            T_jlfunc = llvm::FunctionType::get(T_prjlvalue, ftargs, false);
            assert(T_jlfunc != NULL);
            ftargs.push_back(T_pprjlvalue); // linfo->sparam_vals
            T_jlfuncsparams = llvm::FunctionType::get(T_prjlvalue, ftargs, false);
            assert(T_jlfuncsparams != NULL);

            llvm::Type *vaelts[] = {llvm::PointerType::get(cache.T_int8(context), AddressSpace::Loaded)
                            , cache.T_size(context)
                            , cache.T_int16(context)
                            , cache.T_int16(context)
                            , cache.T_int32(context)
            };
            static_assert(sizeof(jl_array_flags_t) == sizeof(int16_t),
                        "Size of jl_array_flags_t is not the same as int16_t");
            T_jlarray = llvm::StructType::get(context, llvm::makeArrayRef(vaelts));
            T_pjlarray = llvm::PointerType::get(T_jlarray, 0);
        }
    } cache;

    void initialize(llvm::LLVMContext &context) {
        cache.initialize(context, *this);
    }

    //Cached types
    llvm::Type *T_jlvalue(llvm::LLVMContext &context) {
        return cache.T_jlvalue;
    }
    llvm::Type *T_pjlvalue(llvm::LLVMContext &context) {
        return cache.T_pjlvalue;
    }
    llvm::Type *T_prjlvalue(llvm::LLVMContext &context) {
        return cache.T_prjlvalue;
    }
    llvm::Type *T_ppjlvalue(llvm::LLVMContext &context) {
        return cache.T_ppjlvalue;
    }
    llvm::Type *T_pprjlvalue(llvm::LLVMContext &context) {
        return cache.T_pprjlvalue;
    }
    llvm::Type *jl_array_llvmt(llvm::LLVMContext &context) {
        return cache.T_jlarray;
    }
    llvm::Type *jl_parray_llvmt(llvm::LLVMContext &context) {
        return cache.T_pjlarray;
    }
    llvm::FunctionType *jl_func_sig(llvm::LLVMContext &context) {
        return cache.T_jlfunc;
    }
    llvm::FunctionType *jl_func_sig_sparams(llvm::LLVMContext &context) {
        return cache.T_jlfuncsparams;
    }
    llvm::Type *T_pvoidfunc(llvm::LLVMContext &context) {
        return cache.T_pvoidfunc;
    }

    llvm::IntegerType *T_sigatomic(llvm::LLVMContext &context) {
        return llvm::Type::getIntNTy(context, sizeof(sig_atomic_t) * CHAR_BIT);
    }

    //Numeric pointer types

    llvm::Type *T_pint8(llvm::LLVMContext &context) {
        return llvm::Type::getInt8PtrTy(context);
    }
    llvm::Type *T_pint16(llvm::LLVMContext &context) {
        return llvm::Type::getInt16PtrTy(context);
    }
    llvm::Type *T_pint32(llvm::LLVMContext &context) {
        return llvm::Type::getInt32PtrTy(context);
    }
    llvm::Type *T_pint64(llvm::LLVMContext &context) {
        return llvm::Type::getInt64PtrTy(context);
    }
    llvm::Type *T_psize(llvm::LLVMContext &context) {
        return llvm::PointerType::get(T_size(context), 0);
    }
    llvm::Type *T_pfloat32(llvm::LLVMContext &context) {
        return llvm::Type::getFloatPtrTy(context);
    }
    llvm::Type *T_pfloat64(llvm::LLVMContext &context) {
        return llvm::Type::getDoublePtrTy(context);
    }

    llvm::Type *T_ppint8(llvm::LLVMContext &context) {
        return llvm::PointerType::get(T_pint8(context), 0);
    }
    llvm::Type *T_pppint8(llvm::LLVMContext &context) {
        return llvm::PointerType::get(T_ppint8(context), 0);
    }

    //Passthrough types (LLVMContext caches these)

    llvm::Type *T_void(llvm::LLVMContext &context) {
        return llvm::Type::getVoidTy(context);
    }

    llvm::Type *T_float16(llvm::LLVMContext &context) {
        return llvm::Type::getHalfTy(context);
    }
    llvm::Type *T_float32(llvm::LLVMContext &context) {
        return llvm::Type::getFloatTy(context);
    }
    llvm::Type *T_float64(llvm::LLVMContext &context) {
        return llvm::Type::getDoubleTy(context);
    }
    llvm::Type *T_float128(llvm::LLVMContext &context) {
        return llvm::Type::getFP128Ty(context);
    }

    llvm::IntegerType *T_int1(llvm::LLVMContext &context) {
        return llvm::Type::getInt1Ty(context);
    }
    llvm::IntegerType *T_int8(llvm::LLVMContext &context) {
        return llvm::Type::getInt8Ty(context);
    }
    llvm::IntegerType *T_int16(llvm::LLVMContext &context) {
        return llvm::Type::getInt16Ty(context);
    }
    llvm::IntegerType *T_int32(llvm::LLVMContext &context) {
        return llvm::Type::getInt32Ty(context);
    }
    llvm::IntegerType *T_int64(llvm::LLVMContext &context) {
        return llvm::Type::getInt64Ty(context);
    }

    llvm::IntegerType *T_uint8(llvm::LLVMContext &context) {
        return T_int8(context);
    }
    llvm::IntegerType *T_uint16(llvm::LLVMContext &context) {
        return T_int16(context);
    }
    llvm::IntegerType *T_uint32(llvm::LLVMContext &context) {
        return T_int32(context);
    }
    llvm::IntegerType *T_uint64(llvm::LLVMContext &context) {
        return T_int64(context);
    }

    llvm::IntegerType *T_char(llvm::LLVMContext &context) {
        return T_int32(context);
    }
    llvm::IntegerType *T_size(llvm::LLVMContext &context) {
        return sizeof(size_t) > sizeof(uint32_t) ? T_uint64(context) : T_uint32(context);
    }
};

struct MDCache {

    // type-based alias analysis nodes.  Indentation of comments indicates hierarchy.
    llvm::MDNode *tbaa_root;     // Everything
    llvm::MDNode *tbaa_gcframe;    // GC frame
    // LLVM should have enough info for alias analysis of non-gcframe stack slot
    // this is mainly a place holder for `jl_cgval_t::tbaa`
    llvm::MDNode *tbaa_stack;      // stack slot
    llvm::MDNode *tbaa_unionselbyte;   // a selector byte in isbits Union struct fields
    llvm::MDNode *tbaa_data;       // Any user data that `pointerset/ref` are allowed to alias
    llvm::MDNode *tbaa_binding;        // jl_binding_t::value
    llvm::MDNode *tbaa_value;          // jl_value_t, that is not jl_array_t
    llvm::MDNode *tbaa_mutab;              // mutable type
    llvm::MDNode *tbaa_datatype;               // datatype
    llvm::MDNode *tbaa_immut;              // immutable type
    llvm::MDNode *tbaa_ptrarraybuf;    // Data in an array of boxed values
    llvm::MDNode *tbaa_arraybuf;       // Data in an array of POD
    llvm::MDNode *tbaa_array;      // jl_array_t
    llvm::MDNode *tbaa_arrayptr;       // The pointer inside a jl_array_t
    llvm::MDNode *tbaa_arraysize;      // A size in a jl_array_t
    llvm::MDNode *tbaa_arraylen;       // The len in a jl_array_t
    llvm::MDNode *tbaa_arrayflags;     // The flags in a jl_array_t
    llvm::MDNode *tbaa_arrayoffset;     // The offset in a jl_array_t
    llvm::MDNode *tbaa_arrayselbyte;   // a selector byte in a isbits Union jl_array_t
    llvm::MDNode *tbaa_const;      // Memory that is immutable by the time LLVM can see it

    llvm::Attribute Thunk;

    void initialize(llvm::LLVMContext &context) {
        llvm::MDBuilder mbuilder(context);
        llvm::MDNode *jtbaa = mbuilder.createTBAARoot("jtbaa");
        tbaa_root = mbuilder.createTBAAScalarTypeNode("jtbaa", jtbaa);
        auto make_child = [&](const char *name, llvm::MDNode *parent = nullptr, bool isConstant = false){
            llvm::MDNode *scalar = mbuilder.createTBAAScalarTypeNode(name, parent ? parent : tbaa_root);
            llvm::MDNode *n = mbuilder.createTBAAStructTagNode(scalar, scalar, 0, isConstant);
            return std::make_pair(n, scalar);
        };
        tbaa_gcframe = make_child("jtbaa_gcframe").first;
        llvm::MDNode *tbaa_stack_scalar;
        std::tie(tbaa_stack, tbaa_stack_scalar) = make_child("jtbaa_stack");
        tbaa_unionselbyte = make_child("jtbaa_unionselbyte", tbaa_stack_scalar).first;
        llvm::MDNode *tbaa_data_scalar;
        std::tie(tbaa_data, tbaa_data_scalar) = make_child("jtbaa_data");
        tbaa_binding = make_child("jtbaa_binding", tbaa_data_scalar).first;
        llvm::MDNode *tbaa_value_scalar;
        std::tie(tbaa_value, tbaa_value_scalar) =
            make_child("jtbaa_value", tbaa_data_scalar);
        llvm::MDNode *tbaa_mutab_scalar;
        std::tie(tbaa_mutab, tbaa_mutab_scalar) =
            make_child("jtbaa_mutab", tbaa_value_scalar);
        tbaa_datatype = make_child("jtbaa_datatype", tbaa_mutab_scalar).first;
        tbaa_immut = make_child("jtbaa_immut", tbaa_value_scalar).first;
        tbaa_arraybuf = make_child("jtbaa_arraybuf", tbaa_data_scalar).first;
        tbaa_ptrarraybuf = make_child("jtbaa_ptrarraybuf", tbaa_data_scalar).first;
        llvm::MDNode *tbaa_array_scalar;
        std::tie(tbaa_array, tbaa_array_scalar) = make_child("jtbaa_array");
        tbaa_arrayptr = make_child("jtbaa_arrayptr", tbaa_array_scalar).first;
        tbaa_arraysize = make_child("jtbaa_arraysize", tbaa_array_scalar).first;
        tbaa_arraylen = make_child("jtbaa_arraylen", tbaa_array_scalar).first;
        tbaa_arrayflags = make_child("jtbaa_arrayflags", tbaa_array_scalar).first;
        tbaa_arrayoffset = make_child("jtbaa_arrayoffset", tbaa_array_scalar).first;
        tbaa_const = make_child("jtbaa_const", nullptr, true).first;
        tbaa_arrayselbyte = make_child("jtbaa_arrayselbyte", tbaa_array_scalar).first;

        Thunk = llvm::Attribute::get(context, "thunk");
    }
};

struct NullCache {
    llvm::Value *V_null;
    llvm::Value *V_rnull;
    llvm::Value *V_size0;

    void initialize(llvm::LLVMContext &context, TypeCache &types) {
        V_null = llvm::Constant::getNullValue(types.T_pjlvalue(context));
        V_rnull = llvm::Constant::getNullValue(types.T_prjlvalue(context));
        V_size0 = llvm::Constant::getNullValue(types.T_size(context));
    }
};

struct _jl_codegen_params_t;
llvm::DIType *_julia_type_to_di(_jl_codegen_params_t *ctx, jl_value_t *jt, llvm::DIBuilder *dbuilder, bool isboxed);

struct DebugInfoCache {
    llvm::DICompositeType *jl_value_dillvmt;
    llvm::DIDerivedType *jl_pvalue_dillvmt;
    llvm::DIDerivedType *jl_ppvalue_dillvmt;
    llvm::DISubroutineType *jl_di_func_sig;
    llvm::DISubroutineType *jl_di_func_null_sig;

    void initialize(llvm::Module &m) {
        // add needed base debugging definitions to our LLVM environment
        llvm::DIBuilder dbuilder(m);
        llvm::DIFile *julia_h = dbuilder.createFile("julia.h", "");
        jl_value_dillvmt = dbuilder.createStructType(nullptr,
            "jl_value_t",
            julia_h,
            71, // At the time of this writing. Not sure if it's worth it to keep this in sync
            0 * 8, // sizeof(jl_value_t) * 8,
            __alignof__(void*) * 8, // __alignof__(jl_value_t) * 8,
            llvm::DINode::FlagZero, // Flags
            nullptr,    // Derived from
            nullptr);  // Elements - will be corrected later

        jl_pvalue_dillvmt = dbuilder.createPointerType(jl_value_dillvmt, sizeof(jl_value_t*) * 8,
                                                    __alignof__(jl_value_t*) * 8);

        llvm::SmallVector<llvm::Metadata *, 1> Elts;
        std::vector<llvm::Metadata*> diargs(0);
        Elts.push_back(jl_pvalue_dillvmt);
        dbuilder.replaceArrays(jl_value_dillvmt,
        dbuilder.getOrCreateArray(Elts));

        jl_ppvalue_dillvmt = dbuilder.createPointerType(jl_pvalue_dillvmt, sizeof(jl_value_t**) * 8,
                                                        __alignof__(jl_value_t**) * 8);

        diargs.push_back(jl_pvalue_dillvmt);    // Return Type (ret value)
        diargs.push_back(jl_pvalue_dillvmt);    // First Argument (function)
        diargs.push_back(jl_ppvalue_dillvmt);   // Second Argument (argv)
        // Third argument (length(argv))
        diargs.push_back(_julia_type_to_di(NULL, (jl_value_t*)jl_int32_type, &dbuilder, false));

        jl_di_func_sig = dbuilder.createSubroutineType(
            dbuilder.getOrCreateTypeArray(diargs));
        jl_di_func_null_sig = dbuilder.createSubroutineType(
            dbuilder.getOrCreateTypeArray(llvm::None));
    }
};

struct ContextCache {
    llvm::LLVMContext *context;
    TypeCache types;
    MDCache metadata;
    NullCache nulls;
    DebugInfoCache debug;

    void initialize(llvm::LLVMContext &context) {
        this->context = &context;
        types.initialize(context);
        metadata.initialize(context);
        nulls.initialize(context, types);
        //DebugInfoCache remains uninitialized because no module present
    }

    void initialize(llvm::Module &m) {
        initialize(m.getContext());
        debug.initialize(m);
    }

    void checkContext(llvm::LLVMContext &context) {
        assert(this->context == &context && "Initialized and current context are different!");
    }
};

// JLCALL with API arguments ([extra], arg0, arg1, arg2, ...) has the following ABI calling conventions defined:
#define JLCALL_F_CC (CallingConv::ID)37     // (jl_value_t *arg0, jl_value_t **argv, uint32_t nargv)
#define JLCALL_F2_CC (CallingConv::ID)38    // (jl_value_t *arg0, jl_value_t **argv, uint32_t nargv, jl_value_t *extra)

// return how many Tracked pointers are in T (count > 0),
// and if there is anything else in T (all == false)
struct CountTrackedPointers {
    unsigned count = 0;
    bool all = true;
    bool derived = false;
    CountTrackedPointers(llvm::Type *T);
};

unsigned TrackWithShadow(llvm::Value *Src, llvm::Type *T, bool isptr, llvm::Value *Dst, llvm::IRBuilder<> &irbuilder);
std::vector<llvm::Value*> ExtractTrackedValues(llvm::Value *Src, llvm::Type *STy, bool isptr, llvm::IRBuilder<> &irbuilder, llvm::ArrayRef<unsigned> perm_offsets={});

static inline void llvm_dump(llvm::Value *v)
{
    v->print(llvm::dbgs(), true);
    llvm::dbgs() << "\n";
}

static inline void llvm_dump(llvm::Type *v)
{
    v->print(llvm::dbgs(), true);
    llvm::dbgs() << "\n";
}

static inline void llvm_dump(llvm::Function *f)
{
    f->print(llvm::dbgs(), nullptr, false, true);
}

static inline void llvm_dump(llvm::Module *m)
{
    m->print(llvm::dbgs(), nullptr);
}

static inline void llvm_dump(llvm::Metadata *m)
{
    m->print(llvm::dbgs());
    llvm::dbgs() << "\n";
}

static inline void llvm_dump(llvm::DebugLoc *dbg)
{
    dbg->print(llvm::dbgs());
    llvm::dbgs() << "\n";
}

static inline std::pair<llvm::MDNode*,llvm::MDNode*> tbaa_make_child_with_context(llvm::LLVMContext &ctxt, const char *name, llvm::MDNode *parent=nullptr, bool isConstant=false)
{
    llvm::MDBuilder mbuilder(ctxt);
    llvm::MDNode *jtbaa = mbuilder.createTBAARoot("jtbaa");
    llvm::MDNode *tbaa_root = mbuilder.createTBAAScalarTypeNode("jtbaa", jtbaa);
    llvm::MDNode *scalar = mbuilder.createTBAAScalarTypeNode(name, parent ? parent : tbaa_root);
    llvm::MDNode *n = mbuilder.createTBAAStructTagNode(scalar, scalar, 0, isConstant);
    return std::make_pair(n, scalar);
}

static inline llvm::MDNode *get_tbaa_const(llvm::LLVMContext &ctxt) {
    return tbaa_make_child_with_context(ctxt, "jtbaa_const", nullptr, true).first;
}

static inline llvm::Instruction *tbaa_decorate(llvm::MDNode *md, llvm::Instruction *inst)
{
    inst->setMetadata(llvm::LLVMContext::MD_tbaa, md);
    if (llvm::isa<llvm::LoadInst>(inst) && md && md == get_tbaa_const(md->getContext()))
        inst->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(md->getContext(), llvm::None));
    return inst;
}

// bitcast a value, but preserve its address space when dealing with pointer types
static inline llvm::Value *emit_bitcast_with_builder(llvm::IRBuilder<> &builder, llvm::Value *v, llvm::Type *jl_value)
{
    using namespace llvm;
    if (isa<PointerType>(jl_value) &&
        v->getType()->getPointerAddressSpace() != jl_value->getPointerAddressSpace()) {
        // Cast to the proper address space
        Type *jl_value_addr =
                PointerType::get(cast<PointerType>(jl_value)->getElementType(),
                                 v->getType()->getPointerAddressSpace());
        return builder.CreateBitCast(v, jl_value_addr);
    }
    else {
        return builder.CreateBitCast(v, jl_value);
    }
}

// Get PTLS through current task.
static inline llvm::Value *get_current_ptls_from_task(llvm::IRBuilder<> &builder, llvm::Value *current_task, llvm::MDNode *tbaa)
{
    using namespace llvm;
    auto T_ppjlvalue = JuliaType::get_ppjlvalue_ty(builder.getContext());
    auto T_pjlvalue = JuliaType::get_pjlvalue_ty(builder.getContext());
    auto T_size = builder.GetInsertBlock()->getModule()->getDataLayout().getIntPtrType(builder.getContext());
    const int ptls_offset = offsetof(jl_task_t, ptls);
    llvm::Value *pptls = builder.CreateInBoundsGEP(
        T_pjlvalue, current_task,
        ConstantInt::get(T_size, ptls_offset / sizeof(void *)),
        "ptls_field");
    LoadInst *ptls_load = builder.CreateAlignedLoad(T_pjlvalue,
        emit_bitcast_with_builder(builder, pptls, T_ppjlvalue), Align(sizeof(void *)), "ptls_load");
    // Note: Corresponding store (`t->ptls = ptls`) happens in `ctx_switch` of tasks.c.
    tbaa_decorate(tbaa, ptls_load);
    // Using `CastInst::Create` to get an `Instruction*` without explicit cast:
    auto ptls = CastInst::Create(Instruction::BitCast, ptls_load, T_ppjlvalue, "ptls");
    builder.Insert(ptls);
    return ptls;
}

// Compatibility shims for LLVM attribute APIs that were renamed in LLVM 14.
//
// Once we no longer support LLVM < 14, these can be mechanically removed by
// translating foo(Bar, …) into Bar->foo(…) resp. Bar.foo(…).
namespace {
using namespace llvm;

inline void addFnAttr(CallInst *Target, Attribute::AttrKind Attr)
{
#if JL_LLVM_VERSION >= 140000
    Target->addFnAttr(Attr);
#else
    Target->addAttribute(AttributeList::FunctionIndex, Attr);
#endif
}

template<class T, class A>
inline void addRetAttr(T *Target, A Attr)
{
#if JL_LLVM_VERSION >= 140000
    Target->addRetAttr(Attr);
#else
    Target->addAttribute(AttributeList::ReturnIndex, Attr);
#endif
}

inline void addAttributeAtIndex(Function *F, unsigned Index, Attribute Attr)
{
#if JL_LLVM_VERSION >= 140000
    F->addAttributeAtIndex(Index, Attr);
#else
    F->addAttribute(Index, Attr);
#endif
}

inline AttributeSet getFnAttrs(const AttributeList &Attrs)
{
#if JL_LLVM_VERSION >= 140000
    return Attrs.getFnAttrs();
#else
    return Attrs.getFnAttributes();
#endif
}

inline AttributeSet getRetAttrs(const AttributeList &Attrs)
{
#if JL_LLVM_VERSION >= 140000
    return Attrs.getRetAttrs();
#else
    return Attrs.getRetAttributes();
#endif
}

inline bool hasFnAttr(const AttributeList &L, Attribute::AttrKind Kind)
{
#if JL_LLVM_VERSION >= 140000
    return L.hasFnAttr(Kind);
#else
    return L.hasAttribute(AttributeList::FunctionIndex, Kind);
#endif
}

inline AttributeList addAttributeAtIndex(const AttributeList &L, LLVMContext &C,
                                         unsigned Index, Attribute::AttrKind Kind)
{
#if JL_LLVM_VERSION >= 140000
    return L.addAttributeAtIndex(C, Index, Kind);
#else
    return L.addAttribute(C, Index, Kind);
#endif
}

inline AttributeList addAttributeAtIndex(const AttributeList &L, LLVMContext &C,
                                         unsigned Index, Attribute Attr)
{
#if JL_LLVM_VERSION >= 140000
    return L.addAttributeAtIndex(C, Index, Attr);
#else
    return L.addAttribute(C, Index, Attr);
#endif
}

inline AttributeList addAttributesAtIndex(const AttributeList &L, LLVMContext &C,
                                          unsigned Index, const AttrBuilder &Builder)
{
#if JL_LLVM_VERSION >= 140000
    return L.addAttributesAtIndex(C, Index, Builder);
#else
    return L.addAttributes(C, Index, Builder);
#endif
}

inline AttributeList addFnAttribute(const AttributeList &L, LLVMContext &C,
                                    Attribute::AttrKind Kind)
{
#if JL_LLVM_VERSION >= 140000
    return L.addFnAttribute(C, Kind);
#else
    return L.addAttribute(C, AttributeList::FunctionIndex, Kind);
#endif
}

inline AttributeList addRetAttribute(const AttributeList &L, LLVMContext &C,
                                     Attribute::AttrKind Kind)
{
#if JL_LLVM_VERSION >= 140000
    return L.addRetAttribute(C, Kind);
#else
    return L.addAttribute(C, AttributeList::ReturnIndex, Kind);
#endif
}

inline bool hasAttributesAtIndex(const AttributeList &L, unsigned Index)
{
#if JL_LLVM_VERSION >= 140000
    return L.hasAttributesAtIndex(Index);
#else
    return L.hasAttributes(Index);
#endif
}

}
