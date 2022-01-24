// This file is a part of Julia. License is MIT: https://julialang.org/license

#define DEBUG_TYPE "alloc_opt"
#undef DEBUG
#include "llvm-version.h"

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Pass.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/Utils/PromoteMemToReg.h>

#include <llvm/InitializePasses.h>

#include "codegen_shared.h"
#include "julia.h"
#include "julia_internal.h"
#include "llvm-pass-helpers.h"
#include "llvm-alloc-helpers.h"

#include <map>
#include <set>

#include "julia_assert.h"

using namespace llvm;
using namespace jl_alloc;

namespace {

static void removeGCPreserve(CallInst *call, Instruction *val)
{
    auto replace = Constant::getNullValue(val->getType());
    call->replaceUsesOfWith(val, replace);
    for (auto &arg: call->args()) {
        if (!isa<Constant>(arg.get())) {
            return;
        }
    }
    while (!call->use_empty()) {
        auto end = cast<Instruction>(*call->user_begin());
        // gc_preserve_end returns void.
        assert(end->use_empty());
        end->eraseFromParent();
    }
    call->eraseFromParent();
}

/**
 * Promote `julia.gc_alloc_obj` which do not have escaping root to a alloca.
 * Uses that are not considered to escape the object (i.e. heap address) includes,
 *
 * * load
 * * `pointer_from_objref`
 * * Any real llvm intrinsics
 * * gc preserve intrinsics
 * * `ccall` gcroot array (`jl_roots` operand bundle)
 * * store (as address)
 * * addrspacecast, bitcast, getelementptr
 *
 *     The results of these cast instructions will be scanned recursively.
 *
 * All other uses are considered to escape conservatively.
 */

/**
 * TODO:
 * * Return twice
 * * Handle phi node.
 * * Look through `pointer_from_objref`.
 * * Handle jl_box*
 */

struct AllocOpt : public FunctionPass, public JuliaPassContext {
    static char ID;
    AllocOpt()
        : FunctionPass(ID)
    {
        llvm::initializeDominatorTreeWrapperPassPass(*PassRegistry::getPassRegistry());
    }

    const DataLayout *DL;

    Function *lifetime_start;
    Function *lifetime_end;

    Type *T_int64;

private:
    bool doInitialization(Module &m) override;
    bool runOnFunction(Function &F) override;
    void getAnalysisUsage(AnalysisUsage &AU) const override
    {
        FunctionPass::getAnalysisUsage(AU);
        AU.addRequired<DominatorTreeWrapperPass>();
        AU.addPreserved<DominatorTreeWrapperPass>();
        AU.setPreservesCFG();
    }
};

struct Optimizer {
    Optimizer(Function &F, AllocOpt &pass)
        : F(F),
          pass(pass)
    {}

    void initialize();
    void optimizeAll();
    bool finalize();
private:

    void optimizeObject(CallInst *orig, jl_alloc::AllocIdInfo &info);
    void optimizeArray(CallInst *orig, jl_alloc::AllocIdInfo &info);

    bool isSafepoint(Instruction *inst);
    Instruction *getFirstSafepoint(BasicBlock *bb);
    ssize_t getGCAllocSize(Instruction *I);
    void pushInstruction(Instruction *I);

    void insertLifetimeEnd(Value *ptr, Constant *sz, Instruction *insert);
    // insert llvm.lifetime.* calls for `ptr` with size `sz` based on the use of `orig`.
    void insertLifetime(Value *ptr, Constant *sz, Instruction *orig);

    void checkObjectEscapes(Instruction *I);
    bool checkArrayEscapes(Instruction *I);

    void replaceIntrinsicUseWith(IntrinsicInst *call, Intrinsic::ID ID,
                                 Instruction *orig_i, Instruction *new_i);
    void removeAlloc(CallInst *orig_inst, llvm::Value *tag, bool array = false);
    void moveToStack(CallInst *orig_inst, llvm::Value *tag, size_t sz, bool has_ref);
    void splitOnStack(CallInst *orig_inst, llvm::Value *tag);
    void optimizeTag(CallInst *orig_inst, llvm::Value *tag);
    void sinkArrayToErrors(CallInst *orig_inst, size_t bytes);
    void moveArrayToStack(CallInst *orig_inst, jl_alloc::ArrayTypeData &array_type_data, llvm::Value *tag, bool has_ref);
    void fixupArrayAddrSpaces(CallInst *orig_inst, Instruction *arrayshell, Instruction *arraydata, bool has_ref);
    void fixupAddrSpace(Instruction *orig_inst, Instruction *new_inst, bool has_ref);

    Function &F;
    AllocOpt &pass;
    DominatorTree *_DT = nullptr;

    DominatorTree &getDomTree()
    {
        if (!_DT)
            _DT = &pass.getAnalysis<DominatorTreeWrapperPass>().getDomTree();
        return *_DT;
    }
    struct Lifetime {
        struct Frame {
            BasicBlock *bb;
            pred_iterator p_cur;
            pred_iterator p_end;
            Frame(BasicBlock *bb)
                : bb(bb),
                  p_cur(pred_begin(bb)),
                  p_end(pred_end(bb))
            {}
        };
        typedef SmallVector<Frame,4> Stack;
    };
    struct ReplaceUses {
        struct Frame {
            Instruction *orig_i;
            union {
                Instruction *new_i;
                uint32_t offset;
            };
            Frame(Instruction *orig_i, Instruction *new_i)
                : orig_i(orig_i),
                  new_i(new_i)
            {}
            Frame(Instruction *orig_i, uint32_t offset)
                : orig_i(orig_i),
                  offset(offset)
            {}
        };
        typedef SmallVector<Frame,4> Stack;
    };

    std::map<CallInst *, jl_alloc::AllocIdInfo> worklist;
    SmallVector<CallInst*,6> removed;
    AllocUseInfo object_escape_info;
    AllocUseInfo array_escape_info;
    CheckInst::Stack check_stack;
    Lifetime::Stack lifetime_stack;
    ReplaceUses::Stack replace_stack;
    std::map<BasicBlock*, llvm::WeakVH> first_safepoint;
};

void Optimizer::pushInstruction(Instruction *I)
{
    if (auto call = dyn_cast<CallInst>(I)) {
        jl_alloc::AllocIdInfo info;
        if (jl_alloc::getAllocIdInfo(info, call, pass.alloc_obj_func) && (info.isarray || info.object.size != -1)) {
            worklist[call] = std::move(info);
        }
    }
}

void Optimizer::initialize()
{
    for (auto &bb: F) {
        if (isa<UnreachableInst>(bb.getTerminator())) {
            //No point in analyzing allocations that lead to errors
            continue;
        }
        for (auto &I: bb) {
            pushInstruction(&I);
        }
    }
}

void Optimizer::optimizeArray(CallInst *orig, jl_alloc::AllocIdInfo &info) {
    checkObjectEscapes(orig);
    //Upfront remove the typeof calls
    if (object_escape_info.hastypeof) {
        optimizeTag(orig, info.type);
        object_escape_info.hastypeof = false;
    }
    if (object_escape_info.escaped) {
        return;
    }
    if (object_escape_info.refload) {
        //This indicates a load from a
        return;
    }
    assert(!object_escape_info.refstore);
    jl_alloc::ArrayTypeData typeinfo;
    jl_alloc::getArrayType(typeinfo, orig, info);
    //This is going to error anyways, definitely don't bother optimizing
    if (typeinfo.throws_invalid_dims || typeinfo.throws_invalid_size) {
        //TODO consider adding some metadata here identifying this and dealing
        //with the consequences in a later pass?
        return;
    }
    //We can't tell if it's going to throw or not, can't optimize here
    if (typeinfo.dynamic_size || typeinfo.dynamic_type) {
        return;
    }
    //Optimizations on coalescing and hoisting/sinking go here
    if (object_escape_info.returned) {
        return;
    }
    bool may_be_removable = !object_escape_info.addrescaped
                            && (!object_escape_info.haspreserve || !object_escape_info.refstore)
                            && object_escape_info.errors.empty();
    //Totally dead array
    if (may_be_removable && !object_escape_info.hasload) {
        removeAlloc(orig, info.type);
        return;
    }
    bool only_data_pointer = checkArrayEscapes(orig);
    if (array_escape_info.escaped) {
        return;
    }
    if (may_be_removable && only_data_pointer
        && !array_escape_info.hasload && !array_escape_info.refstore) {
        removeAlloc(orig, info.type, true);
        return;
    }
    if (typeinfo.zeroinit) {
        //No GC-tracked pointers for now
        return;
    }
    if (typeinfo.total_size <= ARRAY_INLINE_NBYTES) {
        if (!object_escape_info.errors.empty()) {
            sinkArrayToErrors(orig, typeinfo.total_size);
        }
        moveArrayToStack(orig, typeinfo, info.type, false);
    }
}

void Optimizer::optimizeObject(CallInst *orig, jl_alloc::AllocIdInfo &info) {
    checkObjectEscapes(orig);
    if (object_escape_info.escaped) {
        if (object_escape_info.hastypeof)
            optimizeTag(orig, info.type);
        return;
    }
    //Optimizations on coalescing and hoisting/sinking go here
    if (!object_escape_info.errors.empty() || object_escape_info.returned) {
        if (object_escape_info.hastypeof)
            optimizeTag(orig, info.type);
        return;
    }
    if (!object_escape_info.addrescaped && !object_escape_info.hasload && (!object_escape_info.haspreserve ||
                                                        !object_escape_info.refstore)) {
        // No one took the address, no one reads anything and there's no meaningful
        // preserve of fields (either no preserve/ccall or no object reference fields)
        // We can just delete all the uses.
        removeAlloc(orig, info.type);
        return;
    }
    bool has_ref = false;
    bool has_refaggr = false;
    for (auto memop: object_escape_info.memops) {
        auto &field = memop.second;
        if (field.hasobjref) {
            has_ref = true;
            // This can be relaxed a little based on hasload
            // TODO: add support for hasaggr load/store
            if (field.hasaggr || field.multiloc || field.size != sizeof(void*)) {
                has_refaggr = true;
                break;
            }
        }
    }
    if (!object_escape_info.hasunknownmem && !object_escape_info.addrescaped && !has_refaggr) {
        // No one actually care about the memory layout of this object, split it.
        splitOnStack(orig, info.type);
        return;
    }
    if (has_refaggr) {
        if (object_escape_info.hastypeof)
            optimizeTag(orig, info.type);
        return;
    }
    // The object has no fields with mix reference access
    moveToStack(orig, info.type, info.object.size, has_ref);
}

void Optimizer::optimizeAll()
{
    while (!worklist.empty()) {
        auto it = worklist.begin();
        auto orig = it->first;
        auto info = it->second;
        worklist.erase(it);
        if (info.isarray) {
            optimizeArray(orig, info);
        } else {
            optimizeObject(orig, info);
        }
    }
}

bool Optimizer::finalize()
{
    if (removed.empty())
        return false;
    for (auto inst: removed)
        inst->eraseFromParent();
    return true;
}

bool Optimizer::isSafepoint(Instruction *inst)
{
    auto call = dyn_cast<CallInst>(inst);
    if (!call)
        return false;
    if (isa<IntrinsicInst>(call))
        return false;
    if (auto callee = call->getCalledFunction()) {
        // Known functions emitted in codegen that are not safepoints
        if (callee == pass.pointer_from_objref_func || callee->getName() == "memcmp") {
            return false;
        }
    }
    return true;
}

Instruction *Optimizer::getFirstSafepoint(BasicBlock *bb)
{
    auto it = first_safepoint.find(bb);
    if (it != first_safepoint.end()) {
        Value *Val = it->second;
        if (Val)
            return cast<Instruction>(Val);
    }
    Instruction *first = nullptr;
    for (auto &I: *bb) {
        if (isSafepoint(&I)) {
            first = &I;
            break;
        }
    }
    first_safepoint[bb] = first;
    return first;
}

void Optimizer::checkObjectEscapes(Instruction *I)
{
    object_escape_info.reset();
    jl_alloc::EscapeAnalysisRequiredArgs required{object_escape_info, check_stack, pass, *pass.DL};
    jl_alloc::runEscapeAnalysis(I, required);
}
bool Optimizer::checkArrayEscapes(Instruction *I)
{
    array_escape_info.reset();
    bool only_data_pointer = true;
    jl_alloc::EscapeAnalysisRequiredArgs required{array_escape_info, check_stack, pass, *pass.DL};
    for (auto &memop : object_escape_info.memops) {
        for (auto &access : memop.second.accesses) {
            if (access.offset == offsetof(jl_array_t, data)) {
                jl_alloc::runEscapeAnalysis(access.inst, required);
                assert(!array_escape_info.returned);
                assert(array_escape_info.errors.empty());
                assert(!array_escape_info.hastypeof);
            } else {
                only_data_pointer = false;
            }
        }
    }
    return only_data_pointer;
}

void Optimizer::insertLifetimeEnd(Value *ptr, Constant *sz, Instruction *insert)
{
    BasicBlock::iterator it(insert);
    BasicBlock::iterator begin(insert->getParent()->begin());
    // Makes sure that the end is inserted before nearby start.
    // We insert start before the allocation call, if it is the first safepoint we find for
    // another instruction, it's better if we insert the end before the start instead of the
    // allocation so that the two allocations do not have overlapping lifetime.
    while (it != begin) {
        --it;
        if (auto II = dyn_cast<IntrinsicInst>(&*it)) {
            if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
                II->getIntrinsicID() == Intrinsic::lifetime_end) {
                insert = II;
                continue;
            }
        }
        break;
    }
    CallInst::Create(pass.lifetime_end, {sz, ptr}, "", insert);
}

void Optimizer::insertLifetime(Value *ptr, Constant *sz, Instruction *orig)
{
    CallInst::Create(pass.lifetime_start, {sz, ptr}, "", orig);
    BasicBlock *def_bb = orig->getParent();
    std::set<BasicBlock*> bbs{def_bb};
    auto &DT = getDomTree();
    // Collect all BB where the allocation is live
    for (auto use: object_escape_info.uses) {
        auto bb = use->getParent();
        if (!bbs.insert(bb).second)
            continue;
        assert(lifetime_stack.empty());
        Lifetime::Frame cur{bb};
        while (true) {
            assert(cur.p_cur != cur.p_end);
            auto pred = *cur.p_cur;
            ++cur.p_cur;
            if (bbs.insert(pred).second) {
                if (cur.p_cur != cur.p_end)
                    lifetime_stack.push_back(cur);
                cur = Lifetime::Frame(pred);
            }
            if (cur.p_cur == cur.p_end) {
                if (lifetime_stack.empty())
                    break;
                cur = lifetime_stack.back();
                lifetime_stack.pop_back();
            }
        }
    }
#ifndef JL_NDEBUG
    for (auto bb: bbs) {
        if (bb == def_bb)
            continue;
        if (DT.dominates(orig, bb))
            continue;
        auto F = bb->getParent();
        llvm_dump(F);
        llvm_dump(orig);
        jl_safe_printf("Does not dominate BB:\n");
        llvm_dump(bb);
        abort();
    }
#endif
    // Record extra BBs that contain invisible uses.
    SmallSet<BasicBlock*, 8> extra_use;
    SmallVector<DomTreeNodeBase<BasicBlock>*, 8> dominated;
    for (auto preserve: object_escape_info.preserves) {
        for (auto RN = DT.getNode(preserve->getParent()); RN;
             RN = dominated.empty() ? nullptr : dominated.pop_back_val()) {
            for (auto N: *RN) {
                auto bb = N->getBlock();
                if (extra_use.count(bb))
                    continue;
                bool ended = false;
                for (auto end: preserve->users()) {
                    auto end_bb = cast<Instruction>(end)->getParent();
                    auto end_node = DT.getNode(end_bb);
                    if (end_bb == bb || (end_node && DT.dominates(end_node, N))) {
                        ended = true;
                        break;
                    }
                }
                if (ended)
                    continue;
                bbs.insert(bb);
                extra_use.insert(bb);
                dominated.push_back(N);
            }
        }
        assert(dominated.empty());
    }
    // For each BB, find the first instruction(s) where the allocation is possibly dead.
    // If all successors are live, then there isn't one.
    // If all successors are dead, then it's the first instruction after the last use
    // within the BB.
    // If some successors are live and others are dead, it's the first instruction in
    // the successors that are dead.
    std::vector<Instruction*> first_dead;
    for (auto bb: bbs) {
        bool has_use = false;
        for (auto succ: successors(bb)) {
            // def_bb is the only bb in bbs that's not dominated by orig
            if (succ != def_bb && bbs.count(succ)) {
                has_use = true;
                break;
            }
        }
        if (has_use) {
            for (auto succ: successors(bb)) {
                if (!bbs.count(succ)) {
                    first_dead.push_back(&*succ->begin());
                }
            }
        }
        else if (extra_use.count(bb)) {
            first_dead.push_back(bb->getTerminator());
        }
        else {
            for (auto it = bb->rbegin(), end = bb->rend(); it != end; ++it) {
                if (object_escape_info.uses.count(&*it)) {
                    --it;
                    first_dead.push_back(&*it);
                    break;
                }
            }
        }
    }
    bbs.clear();
    // There can/need only be one lifetime.end for each allocation in each bb, use bbs
    // to record that.
    // Iterate through the first dead and find the first safepoint following each of them.
    while (!first_dead.empty()) {
        auto I = first_dead.back();
        first_dead.pop_back();
        auto bb = I->getParent();
        if (!bbs.insert(bb).second)
            continue;
        if (I == &*bb->begin()) {
            // There's no use in or after this bb. If this bb is not dominated by
            // the def then it has to be dead on entering this bb.
            // Otherwise, there could be use that we don't track
            // before hitting the next safepoint.
            if (!DT.dominates(orig, bb)) {
                insertLifetimeEnd(ptr, sz, &*bb->getFirstInsertionPt());
                continue;
            }
            else if (auto insert = getFirstSafepoint(bb)) {
                insertLifetimeEnd(ptr, sz, insert);
                continue;
            }
        }
        else {
            assert(bb == def_bb || DT.dominates(orig, I));
            BasicBlock::iterator it(I);
            BasicBlock::iterator end = bb->end();
            bool safepoint_found = false;
            for (; it != end; ++it) {
                auto insert = &*it;
                if (isSafepoint(insert)) {
                    insertLifetimeEnd(ptr, sz, insert);
                    safepoint_found = true;
                    break;
                }
            }
            if (safepoint_found) {
                continue;
            }
        }
        for (auto succ: successors(bb)) {
            first_dead.push_back(&*succ->begin());
        }
    }
}

void Optimizer::replaceIntrinsicUseWith(IntrinsicInst *call, Intrinsic::ID ID,
                                        Instruction *orig_i, Instruction *new_i)
{
    auto nargs = call->arg_size();
    SmallVector<Value*, 8> args(nargs);
    SmallVector<Type*, 8> argTys(nargs);
    for (unsigned i = 0; i < nargs; i++) {
        auto arg = call->getArgOperand(i);
        args[i] = arg == orig_i ? new_i : arg;
        argTys[i] = args[i]->getType();
    }
    auto oldfType = call->getFunctionType();
    auto newfType = FunctionType::get(
            oldfType->getReturnType(),
            makeArrayRef(argTys).slice(0, oldfType->getNumParams()),
            oldfType->isVarArg());

    // Accumulate an array of overloaded types for the given intrinsic
    // and compute the new name mangling schema
    SmallVector<Type*, 4> overloadTys;
    {
        SmallVector<Intrinsic::IITDescriptor, 8> Table;
        getIntrinsicInfoTableEntries(ID, Table);
        ArrayRef<Intrinsic::IITDescriptor> TableRef = Table;
        auto res = Intrinsic::matchIntrinsicSignature(newfType, TableRef, overloadTys);
        assert(res == Intrinsic::MatchIntrinsicTypes_Match);
        (void)res;
        bool matchvararg = Intrinsic::matchIntrinsicVarArg(newfType->isVarArg(), TableRef);
        assert(!matchvararg);
        (void)matchvararg;
    }
    auto newF = Intrinsic::getDeclaration(call->getModule(), ID, overloadTys);
    assert(newF->getFunctionType() == newfType);
    newF->setCallingConv(call->getCallingConv());
    auto newCall = CallInst::Create(newF, args, "", call);
    newCall->setTailCallKind(call->getTailCallKind());
    auto old_attrs = call->getAttributes();
    newCall->setAttributes(AttributeList::get(pass.getLLVMContext(), getFnAttrs(old_attrs),
                                              getRetAttrs(old_attrs), {}));
    newCall->setDebugLoc(call->getDebugLoc());
    call->replaceAllUsesWith(newCall);
    call->eraseFromParent();
}

// This function should not erase any safepoint so that the lifetime marker can find and cache
// all the original safepoints.
void Optimizer::moveToStack(CallInst *orig_inst, llvm::Value *tag, size_t sz, bool has_ref)
{
    removed.push_back(orig_inst);
    // The allocation does not escape or get used in a phi node so none of the derived
    // SSA from it are live when we run the allocation again.
    // It is now safe to promote the allocation to an entry block alloca.
    size_t align = 1;
    // TODO: This is overly conservative. May want to instead pass this as a
    //       parameter to the allocation function directly.
    if (sz > 1)
        align = MinAlign(JL_SMALL_BYTE_ALIGNMENT, NextPowerOf2(sz));
    // No debug info for prolog instructions
    IRBuilder<> prolog_builder(&F.getEntryBlock().front());
    AllocaInst *buff;
    Instruction *ptr;
    if (sz == 0) {
        buff = prolog_builder.CreateAlloca(pass.T_int8, ConstantInt::get(pass.T_int64, 0));
        ptr = buff;
    }
    else if (has_ref) {
        // Allocate with the correct type so that the GC frame lowering pass will
        // treat this as a non-mem2reg'd alloca
        // The ccall root and GC preserve handling below makes sure that
        // the alloca isn't optimized out.
        buff = prolog_builder.CreateAlloca(pass.T_prjlvalue);
        buff->setAlignment(Align(align));
        ptr = cast<Instruction>(prolog_builder.CreateBitCast(buff, pass.T_pint8));
    }
    else {
        Type *buffty;
        if (pass.DL->isLegalInteger(sz * 8))
            buffty = Type::getIntNTy(pass.getLLVMContext(), sz * 8);
        else
            buffty = ArrayType::get(Type::getInt8Ty(pass.getLLVMContext()), sz);
        buff = prolog_builder.CreateAlloca(buffty);
        buff->setAlignment(Align(align));
        ptr = cast<Instruction>(prolog_builder.CreateBitCast(buff, pass.T_pint8));
    }
    insertLifetime(ptr, ConstantInt::get(pass.T_int64, sz), orig_inst);
    auto new_inst = cast<Instruction>(prolog_builder.CreateBitCast(ptr, pass.T_pjlvalue));
    new_inst->takeName(orig_inst);

    auto simple_replace = [&] (Instruction *orig_i, Instruction *new_i) {
        if (orig_i->user_empty()) {
            if (orig_i != orig_inst)
                orig_i->eraseFromParent();
            return true;
        }
        Type *orig_t = orig_i->getType();
        Type *new_t = new_i->getType();
        if (orig_t == new_t) {
            orig_i->replaceAllUsesWith(new_i);
            if (orig_i != orig_inst)
                orig_i->eraseFromParent();
            return true;
        }
        return false;
    };
    if (simple_replace(orig_inst, new_inst))
        return;
    assert(replace_stack.empty());
    ReplaceUses::Frame cur{orig_inst, new_inst};
    auto finish_cur = [&] () {
        assert(cur.orig_i->user_empty());
        if (cur.orig_i != orig_inst) {
            cur.orig_i->eraseFromParent();
        }
    };
    auto push_frame = [&] (Instruction *orig_i, Instruction *new_i) {
        if (simple_replace(orig_i, new_i))
            return;
        replace_stack.push_back(cur);
        cur = {orig_i, new_i};
    };
    // Both `orig_i` and `new_i` should be pointer of the same type
    // but possibly different address spaces. `new_i` is always in addrspace 0.
    auto replace_inst = [&] (Instruction *user) {
        Instruction *orig_i = cur.orig_i;
        Instruction *new_i = cur.new_i;
        if (isa<LoadInst>(user) || isa<StoreInst>(user)) {
            user->replaceUsesOfWith(orig_i, new_i);
        }
        else if (auto call = dyn_cast<CallInst>(user)) {
            auto callee = call->getCalledOperand();
            if (pass.pointer_from_objref_func == callee) {
                call->replaceAllUsesWith(new_i);
                call->eraseFromParent();
                return;
            }
            if (pass.typeof_func == callee) {
                call->replaceAllUsesWith(tag);
                call->eraseFromParent();
                return;
            }
            // Also remove the preserve intrinsics so that it can be better optimized.
            if (pass.gc_preserve_begin_func == callee) {
                if (has_ref) {
                    call->replaceUsesOfWith(orig_i, buff);
                }
                else {
                    removeGCPreserve(call, orig_i);
                }
                return;
            }
            if (pass.write_barrier_func == callee) {
                call->eraseFromParent();
                return;
            }
            if (auto intrinsic = dyn_cast<IntrinsicInst>(call)) {
                if (Intrinsic::ID ID = intrinsic->getIntrinsicID()) {
                    replaceIntrinsicUseWith(intrinsic, ID, orig_i, new_i);
                    return;
                }
            }
            // remove from operand bundle
            Value *replace = has_ref ? (Value*)buff : Constant::getNullValue(orig_i->getType());
            user->replaceUsesOfWith(orig_i, replace);
        }
        else if (isa<AddrSpaceCastInst>(user) || isa<BitCastInst>(user)) {
            auto cast_t = PointerType::get(cast<PointerType>(user->getType())->getElementType(),
                                           0);
            auto replace_i = new_i;
            Type *new_t = new_i->getType();
            if (cast_t != new_t) {
                replace_i = new BitCastInst(replace_i, cast_t, "", user);
                replace_i->setDebugLoc(user->getDebugLoc());
                replace_i->takeName(user);
            }
            push_frame(user, replace_i);
        }
        else if (auto gep = dyn_cast<GetElementPtrInst>(user)) {
            SmallVector<Value *, 4> IdxOperands(gep->idx_begin(), gep->idx_end());
            auto new_gep = GetElementPtrInst::Create(gep->getSourceElementType(),
                                                     new_i, IdxOperands,
                                                     gep->getName(), gep);
            new_gep->setIsInBounds(gep->isInBounds());
            new_gep->takeName(gep);
            new_gep->copyMetadata(*gep);
            push_frame(gep, new_gep);
        }
        else {
            abort();
        }
    };

    while (true) {
        replace_inst(cast<Instruction>(*cur.orig_i->user_begin()));
        while (cur.orig_i->use_empty()) {
            finish_cur();
            if (replace_stack.empty())
                return;
            cur = replace_stack.back();
            replace_stack.pop_back();
        }
    }
}

// This function should not erase any safepoint so that the lifetime marker can find and cache
// all the original safepoints.
void Optimizer::removeAlloc(CallInst *orig_inst, llvm::Value *tag, bool array)
{
    removed.push_back(orig_inst);
    auto simple_remove = [&] (Instruction *orig_i) {
        if (orig_i->user_empty()) {
            if (orig_i != orig_inst)
                orig_i->eraseFromParent();
            return true;
        }
        return false;
    };
    if (simple_remove(orig_inst))
        return;
    assert(replace_stack.empty());
    ReplaceUses::Frame cur{orig_inst, nullptr};
    auto finish_cur = [&] () {
        assert(cur.orig_i->user_empty());
        if (cur.orig_i != orig_inst) {
            cur.orig_i->eraseFromParent();
        }
    };
    auto push_frame = [&] (Instruction *orig_i) {
        if (simple_remove(orig_i))
            return;
        replace_stack.push_back(cur);
        cur = {orig_i, nullptr};
    };
    auto remove_inst = [&] (Instruction *user) {
        Instruction *orig_i = cur.orig_i;
        if (auto store = dyn_cast<StoreInst>(user)) {
            // All stores are known to be dead.
            // The stored value might be an gc pointer in which case deleting the object
            // might open more optimization opportunities.
            if (auto stored_inst = dyn_cast<Instruction>(store->getValueOperand()))
                pushInstruction(stored_inst);
            user->eraseFromParent();
            return;
        }
        else if (auto call = dyn_cast<CallInst>(user)) {
            auto callee = call->getCalledOperand();
            if (pass.gc_preserve_begin_func == callee) {
                removeGCPreserve(call, orig_i);
                return;
            }
            if (pass.typeof_func == callee) {
                call->replaceAllUsesWith(tag);
                call->eraseFromParent();
                return;
            }
            if (pass.write_barrier_func == callee) {
                call->eraseFromParent();
                return;
            }
            if (auto II = dyn_cast<IntrinsicInst>(call)) {
                auto id = II->getIntrinsicID();
                if (id == Intrinsic::memset || id == Intrinsic::lifetime_start ||
                    id == Intrinsic::lifetime_end || isa<DbgInfoIntrinsic>(II)) {
                    call->eraseFromParent();
                    return;
                }
            }
            // remove from operand bundle
            user->replaceUsesOfWith(orig_i, Constant::getNullValue(orig_i->getType()));
        }
        else if (isa<AddrSpaceCastInst>(user) || isa<BitCastInst>(user) ||
                 isa<GetElementPtrInst>(user)) {
            push_frame(user);
        }
        else if (array && isa<LoadInst>(user)) {
            push_frame(user);
        }
        else {
            abort();
        }
    };

    while (true) {
        remove_inst(cast<Instruction>(*cur.orig_i->user_begin()));
        while (cur.orig_i->use_empty()) {
            finish_cur();
            if (replace_stack.empty())
                return;
            cur = replace_stack.back();
            replace_stack.pop_back();
        }
    }
}

// Unable to optimize out the allocation, do store to load forwarding on the tag instead.
void Optimizer::optimizeTag(CallInst *orig_inst, llvm::Value *tag)
{
    // `julia.typeof` is only legal on the original pointer, no need to scan recursively
    size_t last_deleted = removed.size();
    for (auto user: orig_inst->users()) {
        if (auto call = dyn_cast<CallInst>(user)) {
            auto callee = call->getCalledOperand();
            if (pass.typeof_func == callee) {
                call->replaceAllUsesWith(tag);
                // Push to the removed instructions to trigger `finalize` to
                // return the correct result.
                // Also so that we don't have to worry about iterator invalidation...
                removed.push_back(call);
            }
        }
    }
    while (last_deleted < removed.size())
        removed[last_deleted++]->replaceUsesOfWith(orig_inst, UndefValue::get(orig_inst->getType()));
}

void Optimizer::splitOnStack(CallInst *orig_inst, llvm::Value *tag)
{
    removed.push_back(orig_inst);
    IRBuilder<> prolog_builder(&F.getEntryBlock().front());
    struct SplitSlot {
        AllocaInst *slot;
        bool isref;
        uint32_t offset;
        uint32_t size;
    };
    SmallVector<SplitSlot,8> slots;
    for (auto memop: object_escape_info.memops) {
        auto offset = memop.first;
        auto &field = memop.second;
        // If the field has no reader and is not a object reference field that we
        // need to preserve at some point, there's no need to allocate the field.
        if (!field.hasload && (!field.hasobjref || !object_escape_info.haspreserve))
            continue;
        SplitSlot slot{nullptr, field.hasobjref, offset, field.size};
        Type *allocty;
        if (field.hasobjref) {
            allocty = pass.T_prjlvalue;
        }
        else if (field.elty && !field.multiloc) {
            allocty = field.elty;
        }
        else if (pass.DL->isLegalInteger(field.size * 8)) {
            allocty = Type::getIntNTy(pass.getLLVMContext(), field.size * 8);
        } else {
            allocty = ArrayType::get(Type::getInt8Ty(pass.getLLVMContext()), field.size);
        }
        slot.slot = prolog_builder.CreateAlloca(allocty);
        insertLifetime(prolog_builder.CreateBitCast(slot.slot, pass.T_pint8),
                       ConstantInt::get(pass.T_int64, field.size), orig_inst);
        slots.push_back(std::move(slot));
    }
    const auto nslots = slots.size();
    auto find_slot = [&] (uint32_t offset) {
        if (offset == 0)
            return 0u;
        unsigned lb = 0;
        unsigned ub = slots.size();
        while (lb + 1 < ub) {
            unsigned mid = (lb + ub) / 2;
            if (slots[mid].offset <= offset) {
                lb = mid;
            }
            else {
                ub = mid;
            }
        }
        return lb;
    };
    auto simple_replace = [&] (Instruction *orig_i) {
        if (orig_i->user_empty()) {
            if (orig_i != orig_inst)
                orig_i->eraseFromParent();
            return true;
        }
        return false;
    };
    if (simple_replace(orig_inst))
        return;
    assert(replace_stack.empty());
    ReplaceUses::Frame cur{orig_inst, uint32_t(0)};
    auto finish_cur = [&] () {
        assert(cur.orig_i->user_empty());
        if (cur.orig_i != orig_inst) {
            cur.orig_i->eraseFromParent();
        }
    };
    auto push_frame = [&] (Instruction *orig_i, uint32_t offset) {
        if (simple_replace(orig_i))
            return;
        replace_stack.push_back(cur);
        cur = {orig_i, offset};
    };
    auto slot_gep = [&] (SplitSlot &slot, uint32_t offset, Type *elty, IRBuilder<> &builder) {
        assert(slot.offset <= offset);
        offset -= slot.offset;
        auto size = pass.DL->getTypeAllocSize(elty);
        Value *addr;
        if (offset % size == 0) {
            addr = builder.CreateBitCast(slot.slot, elty->getPointerTo());
            if (offset != 0) {
                addr = builder.CreateConstInBoundsGEP1_32(elty, addr, offset / size);
            }
        }
        else {
            addr = builder.CreateBitCast(slot.slot, pass.T_pint8);
            addr = builder.CreateConstInBoundsGEP1_32(pass.T_int8, addr, offset);
            addr = builder.CreateBitCast(addr, elty->getPointerTo());
        }
        return addr;
    };
    auto replace_inst = [&] (Use *use) {
        Instruction *user = cast<Instruction>(use->getUser());
        Instruction *orig_i = cur.orig_i;
        uint32_t offset = cur.offset;
        if (auto load = dyn_cast<LoadInst>(user)) {
            auto slot_idx = find_slot(offset);
            auto &slot = slots[slot_idx];
            assert(slot.offset <= offset && slot.offset + slot.size >= offset);
            IRBuilder<> builder(load);
            Value *val;
            Type *load_ty = load->getType();
            LoadInst *newload;
            if (slot.isref) {
                assert(slot.offset == offset);
                newload = builder.CreateLoad(pass.T_prjlvalue, slot.slot);
                // Assume the addrspace is correct.
                val = builder.CreateBitCast(newload, load_ty);
            }
            else {
                newload = builder.CreateLoad(load_ty, slot_gep(slot, offset, load_ty, builder));
                val = newload;
            }
            // TODO: should we use `load->clone()`, or manually copy any other metadata?
            newload->setAlignment(load->getAlign());
            // since we're moving heap-to-stack, it is safe to downgrade the atomic level to NotAtomic
            newload->setOrdering(AtomicOrdering::NotAtomic);
            load->replaceAllUsesWith(val);
            load->eraseFromParent();
            return;
        }
        else if (auto store = dyn_cast<StoreInst>(user)) {
            if (auto stored_inst = dyn_cast<Instruction>(store->getValueOperand()))
                pushInstruction(stored_inst);
            auto slot_idx = find_slot(offset);
            auto &slot = slots[slot_idx];
            if (slot.offset > offset || slot.offset + slot.size <= offset) {
                store->eraseFromParent();
                return;
            }
            IRBuilder<> builder(store);
            auto store_val = store->getValueOperand();
            auto store_ty = store_val->getType();
            StoreInst *newstore;
            if (slot.isref) {
                assert(slot.offset == offset);
                if (!isa<PointerType>(store_ty)) {
                    store_val = builder.CreateBitCast(store_val, pass.T_size);
                    store_val = builder.CreateIntToPtr(store_val, pass.T_pjlvalue);
                    store_ty = pass.T_pjlvalue;
                }
                else {
                    store_ty = cast<PointerType>(pass.T_pjlvalue)->getElementType()
                        ->getPointerTo(cast<PointerType>(store_ty)->getAddressSpace());
                    store_val = builder.CreateBitCast(store_val, store_ty);
                }
                if (cast<PointerType>(store_ty)->getAddressSpace() != AddressSpace::Tracked)
                    store_val = builder.CreateAddrSpaceCast(store_val, pass.T_prjlvalue);
                newstore = builder.CreateStore(store_val, slot.slot);
            }
            else {
                newstore = builder.CreateStore(store_val, slot_gep(slot, offset, store_ty, builder));
            }
            // TODO: should we use `store->clone()`, or manually copy any other metadata?
            newstore->setAlignment(store->getAlign());
            // since we're moving heap-to-stack, it is safe to downgrade the atomic level to NotAtomic
            newstore->setOrdering(AtomicOrdering::NotAtomic);
            store->eraseFromParent();
            return;
        }
        else if (isa<AtomicCmpXchgInst>(user) || isa<AtomicRMWInst>(user)) {
            auto slot_idx = find_slot(offset);
            auto &slot = slots[slot_idx];
            assert(slot.offset <= offset && slot.offset + slot.size >= offset);
            IRBuilder<> builder(user);
            Value *newptr;
            if (slot.isref) {
                assert(slot.offset == offset);
                newptr = slot.slot;
            }
            else {
                Value *Val = isa<AtomicCmpXchgInst>(user) ? cast<AtomicCmpXchgInst>(user)->getNewValOperand() : cast<AtomicRMWInst>(user)->getValOperand();
                newptr = slot_gep(slot, offset, Val->getType(), builder);
            }
            *use = newptr;
        }
        else if (auto call = dyn_cast<CallInst>(user)) {
            auto callee = call->getCalledOperand();
            assert(callee); // makes it clear for clang analyser that `callee` is not NULL
            if (auto intrinsic = dyn_cast<IntrinsicInst>(call)) {
                if (Intrinsic::ID id = intrinsic->getIntrinsicID()) {
                    if (id == Intrinsic::memset) {
                        IRBuilder<> builder(call);
                        auto val_arg = cast<ConstantInt>(call->getArgOperand(1));
                        auto size_arg = cast<ConstantInt>(call->getArgOperand(2));
                        uint8_t val = val_arg->getLimitedValue();
                        uint32_t size = size_arg->getLimitedValue();
                        for (unsigned idx = find_slot(offset); idx < nslots; idx++) {
                            auto &slot = slots[idx];
                            if (slot.offset + slot.size <= offset || slot.offset >= offset + size)
                                break;
                            if (slot.isref) {
                                assert(slot.offset >= offset &&
                                       slot.offset + slot.size <= offset + size);
                                Constant *ptr;
                                if (val == 0) {
                                    ptr = Constant::getNullValue(pass.T_prjlvalue);
                                }
                                else {
                                    uint64_t intval;
                                    memset(&intval, val, 8);
                                    Constant *val = ConstantInt::get(pass.T_size, intval);
                                    val = ConstantExpr::getIntToPtr(val, pass.T_pjlvalue);
                                    ptr = ConstantExpr::getAddrSpaceCast(val, pass.T_prjlvalue);
                                }
                                StoreInst *store = builder.CreateAlignedStore(ptr, slot.slot, Align(sizeof(void*)));
                                store->setOrdering(AtomicOrdering::NotAtomic);
                                continue;
                            }
                            auto ptr8 = builder.CreateBitCast(slot.slot, pass.T_pint8);
                            if (offset > slot.offset)
                                ptr8 = builder.CreateConstInBoundsGEP1_32(pass.T_int8, ptr8,
                                                                          offset - slot.offset);
                            auto sub_size = std::min(slot.offset + slot.size, offset + size) -
                                std::max(offset, slot.offset);
                            // TODO: alignment computation
                            builder.CreateMemSet(ptr8, val_arg, sub_size, MaybeAlign(0));
                        }
                        call->eraseFromParent();
                        return;
                    }
                    call->eraseFromParent();
                    return;
                }
            }
            if (pass.typeof_func == callee) {
                call->replaceAllUsesWith(tag);
                call->eraseFromParent();
                return;
            }
            if (pass.write_barrier_func == callee) {
                call->eraseFromParent();
                return;
            }
            if (pass.gc_preserve_begin_func == callee) {
                SmallVector<Value*,8> operands;
                for (auto &arg: call->args()) {
                    if (arg.get() == orig_i || isa<Constant>(arg.get()))
                        continue;
                    operands.push_back(arg.get());
                }
                IRBuilder<> builder(call);
                for (auto &slot: slots) {
                    if (!slot.isref)
                        continue;
                    LoadInst *ref = builder.CreateAlignedLoad(pass.T_prjlvalue, slot.slot, Align(sizeof(void*)));
                    // since we're moving heap-to-stack, it is safe to downgrade the atomic level to NotAtomic
                    ref->setOrdering(AtomicOrdering::NotAtomic);
                    operands.push_back(ref);
                }
                auto new_call = builder.CreateCall(pass.gc_preserve_begin_func, operands);
                new_call->takeName(call);
                new_call->setAttributes(call->getAttributes());
                call->replaceAllUsesWith(new_call);
                call->eraseFromParent();
                return;
            }
            // remove from operand bundle
            assert(call->isBundleOperand(use->getOperandNo()));
            assert(call->getOperandBundleForOperand(use->getOperandNo()).getTagName() ==
                   "jl_roots");
            SmallVector<OperandBundleDef,2> bundles;
            call->getOperandBundlesAsDefs(bundles);
            for (auto &bundle: bundles) {
                if (bundle.getTag() != "jl_roots")
                    continue;
                std::vector<Value*> operands;
                for (auto op: bundle.inputs()) {
                    if (op == orig_i || isa<Constant>(op))
                        continue;
                    operands.push_back(op);
                }
                IRBuilder<> builder(call);
                for (auto &slot: slots) {
                    if (!slot.isref)
                        continue;
                    LoadInst *ref = builder.CreateAlignedLoad(pass.T_prjlvalue, slot.slot, Align(sizeof(void*)));
                    // since we're moving heap-to-stack, it is safe to downgrade the atomic level to NotAtomic
                    ref->setOrdering(AtomicOrdering::NotAtomic);
                    operands.push_back(ref);
                }
                bundle = OperandBundleDef("jl_roots", std::move(operands));
                break;
            }
            auto new_call = CallInst::Create(call, bundles, call);
            new_call->takeName(call);
            call->replaceAllUsesWith(new_call);
            call->eraseFromParent();
            return;
        }
        else if (isa<AddrSpaceCastInst>(user) || isa<BitCastInst>(user)) {
            push_frame(user, offset);
        }
        else if (auto gep = dyn_cast<GetElementPtrInst>(user)) {
            APInt apoffset(sizeof(void*) * 8, offset, true);
            gep->accumulateConstantOffset(*pass.DL, apoffset);
            push_frame(gep, apoffset.getLimitedValue());
        }
        else {
            abort();
        }
    };

    while (true) {
        replace_inst(&*cur.orig_i->use_begin());
        while (cur.orig_i->use_empty()) {
            finish_cur();
            if (replace_stack.empty())
                goto cleanup;
            cur = replace_stack.back();
            replace_stack.pop_back();
        }
    }
cleanup:
    for (auto &slot: slots) {
        if (!slot.isref)
            continue;
        PromoteMemToReg({slot.slot}, getDomTree());
    }
}

void Optimizer::moveArrayToStack(CallInst *orig_inst, jl_alloc::ArrayTypeData &array_type_data, llvm::Value *tag, bool has_ref) {
    // We take a simple approach to moving arrays to the stack
    // 1. Create the outer jl_array_t shell struct
    // (copied here for convenience)
    /* JL_EXTENSION typedef struct {
        JL_DATA_TYPE
        void *data;
        size_t length;
        jl_array_flags_t flags;
        uint16_t elsize;  // element size including alignment (dim 1 memory stride)
        uint32_t offset;  // for 1-d only. does not need to get big.
        size_t nrows;
        union {
            // 1d
            size_t maxsize;
            // Nd
            size_t ncols;
        };
        // other dim sizes go here for ndims > 2
        // followed by alignment padding and inline data, or owner pointer
    } jl_array_t; */
    // 2. Populate each of the fields of the struct as if it had been allocated
    // 3. Mark any load memops on the shell with invariant group metadata because all of the fields are constant
    // 4. RAUW on orig_inst and delete orig_inst

    // Step 0: setup
    auto ndimwords = jl_array_ndimwords(orig_inst->arg_size() - 1);
    IRBuilder<> array_builder(&*orig_inst->getFunction()->getEntryBlock().getFirstInsertionPt());
    auto tsz = sizeof(jl_array_t) + ndimwords * sizeof(size_t);
    //Needed for flags.pooled computation below
    if (array_type_data.total_size >= ARRAY_CACHE_ALIGN_THRESHOLD)
        tsz = LLT_ALIGN(tsz, JL_CACHE_BYTE_ALIGNMENT);
    else if (array_type_data.isunboxed && array_type_data.elsz >= 4)
        tsz = LLT_ALIGN(tsz, JL_SMALL_BYTE_ALIGNMENT);
    // Step 1: array shell
    auto arrayshellbacking = array_builder.CreateAlloca(pass.T_int8, ConstantInt::get(pass.T_size, sizeof(void*) + tsz));
    arrayshellbacking->setAlignment(Align(alignof(jl_array_t)));
    auto arrayshell = cast<Instruction>(array_builder.CreateInBoundsGEP(pass.T_int8, arrayshellbacking, ConstantInt::get(pass.T_size, sizeof(void*))));
    arrayshell->takeName(orig_inst);
    arrayshell->setDebugLoc(orig_inst->getDebugLoc());

    //Step 1.5: array type tag
    auto type_ptr = array_builder.CreatePointerCast(arrayshellbacking, PointerType::get(tag->getType(), 0));
    array_builder.CreateAlignedStore(tag, type_ptr, Align(alignof(void*)));

    //Step 2: initialize shell fields
#define initialize(field, value) \
    do { auto field = value; \
    auto field##ptr = array_builder.CreatePointerCast(array_builder.CreateInBoundsGEP(pass.T_int8, arrayshell, ConstantInt::get(pass.T_size, offsetof(jl_array_t, field))), PointerType::get(field->getType(), 0), arrayshell->getName() + ("." #field ".ptr")); \
    array_builder.CreateAlignedStore(field, field##ptr, Align(alignof(decltype(jl_array_t::field)))); } while (0)

    // Step 2a: array data
    auto arraydata = array_builder.CreateAlloca(pass.T_int8, ConstantInt::get(pass.T_size, array_type_data.total_size));
    //TODO do we want to increase our alloca alignment to cache line size if it's big enough or not?
    arraydata->setAlignment(Align(llvm::MinAlign(1, array_type_data.align)));
    arraydata->setName(arrayshell->getName() + ".data");
    if (array_type_data.zeroinit) {
        array_builder.CreateMemSet(arraydata, Constant::getNullValue(pass.T_int8), arraydata->getArraySize(), arraydata->getAlign());
    }
    initialize(data, arraydata);
    // Step 2b: length
    initialize(length, ConstantInt::get(pass.T_size, array_type_data.numels));
    // Step 2c: flags
    jl_array_flags_t flags;
    static_assert(sizeof(jl_array_flags_t) == sizeof(uint16_t), "Expected jl_array_flags_t to be a uint16_t!");
    flags.how = 0; // Known b/c we only stack allocate arrays <= ARRAY_INLINE_NBYTES
    flags.pooled = tsz + array_type_data.total_size <= GC_MAX_SZCLASS; // Unsure if this is actually needed
    flags.ndims = orig_inst->arg_size() - 1;
    flags.ptrarray = !array_type_data.isunboxed;
    flags.hasptr = array_type_data.hasptr;
    flags.isshared = false;
    flags.isaligned = true;
    uint16_t flagsint;
    memcpy(&flagsint, &flags, sizeof(flags)); // TBAA-safe reinterpret
    initialize(flags, ConstantInt::get(Type::getInt16Ty(orig_inst->getContext()), flagsint));
    // Step 2d: elsize
    initialize(elsize, ConstantInt::get(Type::getInt16Ty(orig_inst->getContext()), array_type_data.elsz));
    // Step 2e: offset
    initialize(offset, Constant::getNullValue(pass.T_int32));
    // Step 2f: nrows
    auto dim1 = array_builder.CreateIntCast(orig_inst->getArgOperand(1), pass.T_size, false);
    initialize(nrows, dim1);
    // Step 2g: maxsize/ncols
    if (orig_inst->arg_size() == 2) {
        initialize(maxsize, dim1);
    } else {
        initialize(ncols, array_builder.CreateIntCast(orig_inst->getArgOperand(2), pass.T_size, false));
    }
    // Step 2h: dim3
    if (orig_inst->arg_size() == 4) {
        auto dim3 = array_builder.CreateIntCast(orig_inst->getArgOperand(3), pass.T_size, false);
        auto dim3ptr = array_builder.CreatePointerCast(array_builder.CreateInBoundsGEP(pass.T_int8, arrayshell, ConstantInt::get(pass.T_size, sizeof(jl_array_t))), PointerType::get(dim3->getType(), 0));
        array_builder.CreateAlignedStore(dim3, dim3ptr, Align(alignof(size_t)));
    }
#undef initialize

    // Step 3: invariant group metadata
    //TODO

    // Step 4: RAUW and delete
    //Fixup address spaces
    fixupArrayAddrSpaces(orig_inst, arrayshell, arraydata, has_ref);
    removeAlloc(orig_inst, tag, true);
}

// Replaces arraydata and fixes address spaces for arrayshell and arraydata to addrspace 0
void Optimizer::fixupArrayAddrSpaces(CallInst *orig_inst, Instruction *arrayshell, Instruction *arraydata, bool has_ref) {
    IRBuilder<> builder(orig_inst->getContext());
    for (auto &memop : object_escape_info.memops) {
        for (auto &access : memop.second.accesses) {
            if (access.offset == 0) {
                // This is a data pointer, will have addrspace 13
                // Need to correct to addrspace 0
                builder.SetInsertPoint(access.inst);
                auto casted = builder.CreatePointerCast(arraydata, PointerType::get(cast<PointerType>(access.inst->getType())->getElementType(), 0));
                if (casted != arraydata) {
                    casted->takeName(access.inst);
                    cast<Instruction>(casted)->setDebugLoc(access.inst->getDebugLoc());
                }
                fixupAddrSpace(access.inst, cast<Instruction>(casted), has_ref);
            }
        }
    }
    fixupAddrSpace(orig_inst, arrayshell, has_ref);
}

void Optimizer::fixupAddrSpace(Instruction *orig_inst, Instruction *new_inst, bool has_ref) {
    auto simple_replace = [&] (Instruction *orig_i, Instruction *new_i) {
        if (orig_i->user_empty()) {
            if (orig_i != orig_inst)
                orig_i->eraseFromParent();
            return true;
        }
        Type *orig_t = orig_i->getType();
        Type *new_t = new_i->getType();
        if (orig_t == new_t) {
            orig_i->replaceAllUsesWith(new_i);
            if (orig_i != orig_inst)
                orig_i->eraseFromParent();
            return true;
        }
        return false;
    };
    if (simple_replace(orig_inst, new_inst))
        return;
    assert(replace_stack.empty());
    ReplaceUses::Frame cur{orig_inst, new_inst};
    auto finish_cur = [&] () {
        assert(cur.orig_i->user_empty());
        if (cur.orig_i != orig_inst) {
            cur.orig_i->eraseFromParent();
        }
    };
    auto push_frame = [&] (Instruction *orig_i, Instruction *new_i) {
        if (simple_replace(orig_i, new_i))
            return;
        replace_stack.push_back(cur);
        cur = {orig_i, new_i};
    };
    // Both `orig_i` and `new_i` should be pointer of the same type
    // but possibly different address spaces. `new_i` is always in addrspace 0.
    auto replace_inst = [&] (Instruction *user) {
        Instruction *orig_i = cur.orig_i;
        Instruction *new_i = cur.new_i;
        if (isa<LoadInst>(user) || isa<StoreInst>(user)) {
            user->replaceUsesOfWith(orig_i, new_i);
        }
        else if (auto call = dyn_cast<CallInst>(user)) {
            auto callee = call->getCalledOperand();
            if (pass.pointer_from_objref_func == callee) {
                call->replaceAllUsesWith(new_i);
                call->eraseFromParent();
                return;
            }
            // Also remove the preserve intrinsics so that it can be better optimized.
            if (pass.gc_preserve_begin_func == callee) {
                if (has_ref) {
                    call->replaceUsesOfWith(orig_i, new_inst);
                }
                else {
                    removeGCPreserve(call, orig_i);
                }
                return;
            }
            if (pass.write_barrier_func == callee) {
                call->eraseFromParent();
                return;
            }
            if (auto intrinsic = dyn_cast<IntrinsicInst>(call)) {
                if (Intrinsic::ID ID = intrinsic->getIntrinsicID()) {
                    replaceIntrinsicUseWith(intrinsic, ID, orig_i, new_i);
                    return;
                }
            }
            // remove from operand bundle
            Value *replace = has_ref ? (Value*)new_inst : Constant::getNullValue(orig_i->getType());
            user->replaceUsesOfWith(orig_i, replace);
        }
        else if (isa<AddrSpaceCastInst>(user) || isa<BitCastInst>(user)) {
            auto cast_t = PointerType::get(cast<PointerType>(user->getType())->getElementType(),
                                           0);
            auto replace_i = new_i;
            Type *new_t = new_i->getType();
            if (cast_t != new_t) {
                replace_i = new BitCastInst(replace_i, cast_t, "", user);
                replace_i->setDebugLoc(user->getDebugLoc());
                replace_i->takeName(user);
            }
            push_frame(user, replace_i);
        }
        else if (auto gep = dyn_cast<GetElementPtrInst>(user)) {
            SmallVector<Value *, 4> IdxOperands(gep->idx_begin(), gep->idx_end());
            auto new_gep = GetElementPtrInst::Create(gep->getSourceElementType(),
                                                     new_i, IdxOperands,
                                                     gep->getName(), gep);
            new_gep->setIsInBounds(gep->isInBounds());
            new_gep->takeName(gep);
            new_gep->copyMetadata(*gep);
            push_frame(gep, new_gep);
        }
        else {
            abort();
        }
    };

    while (true) {
        replace_inst(cast<Instruction>(*cur.orig_i->user_begin()));
        while (cur.orig_i->use_empty()) {
            finish_cur();
            if (replace_stack.empty())
                return;
            cur = replace_stack.back();
            replace_stack.pop_back();
        }
    }
}

void Optimizer::sinkArrayToErrors(CallInst *orig, size_t bytes) {
    //The array is now being allocated only at error sites
    //Assuming we don't care about performance of errors,
    //we can be rather wasteful in ensuring that the
    //observable behavior of the program doesn't change
    for (auto &bb : object_escape_info.errors) {
        Instruction *insert = &*bb.first->getFirstInsertionPt();
        IRBuilder<> builder(insert);
        llvm::SmallVector<Value *, 4> args(orig->arg_begin(), orig->arg_end());
        auto copy = builder.CreateCall(orig->getFunctionType(), orig->getCalledOperand(), args);
        copy->setAttributes(orig->getAttributes());
        copy->copyMetadata(*orig);
        copy->setName(orig->getName());
        //Not bothering with metadata here, it's going to be pointless once LLVM sees
        //the original buffer being allocated on the stack and forwards the load
        auto get_data_ptr = [&](Instruction *array, AddressSpace as) {
            return builder.CreateAlignedLoad(PointerType::get(Type::getInt8Ty(array->getContext()), as),
                    builder.CreateBitOrPointerCast(array,
                        PointerType::get(PointerType::get(Type::getInt8Ty(array->getContext()), as),
                            cast<PointerType>(array->getType())->getAddressSpace())),
                    Align(alignof(void*)));
        };
        auto orig_data = get_data_ptr(orig, AddressSpace::Generic);
        auto new_data = get_data_ptr(copy, AddressSpace::Loaded);
        builder.CreateMemCpy(new_data, Align(1), orig_data, Align(1), ConstantInt::get(pass.T_size, bytes));
        for (auto &inst : *bb.first) {
            if (object_escape_info.uses.contains(&inst)) {
                inst.replaceUsesOfWith(orig, copy);
                object_escape_info.uses.erase(&inst);
            }
        }
    }
    object_escape_info.errors.clear();
}

bool AllocOpt::doInitialization(Module &M)
{
    initAll(M);

    DL = &M.getDataLayout();

    T_int64 = Type::getInt64Ty(getLLVMContext());

    lifetime_start = Intrinsic::getDeclaration(&M, Intrinsic::lifetime_start, { T_pint8 });
    lifetime_end = Intrinsic::getDeclaration(&M, Intrinsic::lifetime_end, { T_pint8 });

    return true;
}

bool AllocOpt::runOnFunction(Function &F)
{
    Optimizer optimizer(F, *this);
    optimizer.initialize();
    optimizer.optimizeAll();
    return optimizer.finalize();
}

char AllocOpt::ID = 0;
static RegisterPass<AllocOpt> X("AllocOpt", "Promote heap allocation to stack",
                                false /* Only looks at CFG */,
                                false /* Analysis Pass */);

}

Pass *createAllocOptPass()
{
    return new AllocOpt();
}

extern "C" JL_DLLEXPORT void LLVMExtraAddAllocOptPass_impl(LLVMPassManagerRef PM)
{
    unwrap(PM)->add(createAllocOptPass());
}
