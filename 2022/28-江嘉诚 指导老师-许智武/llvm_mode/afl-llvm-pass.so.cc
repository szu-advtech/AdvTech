/*
  Copyright 2015 Google LLC All rights reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

/*
   american fuzzy lop - LLVM-mode instrumentation pass
   ---------------------------------------------------

   Written by Laszlo Szekeres <lszekeres@google.com> and
              Michal Zalewski <lcamtuf@google.com>

   LLVM integration design comes from Laszlo Szekeres. C bits copied-and-pasted
   from afl-as.c are Michal's fault.

   This library is plugged into LLVM when invoking clang through afl-clang-fast.
   It tells the compiler to add code roughly equivalent to the bits discussed
   in ../afl-as.h.
*/

#define AFL_LLVM_PASS

#include "../config.h"
#include "../debug.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <array>
#include <vector>
#include <map>
#include <set>

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "llvm/IR/CFG.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include<fstream>
#include <string>

using namespace llvm;

using namespace std;

namespace
{

  class AFLCoverage : public ModulePass
  {

  public:
    static char ID;
    AFLCoverage() : ModulePass(ID) {}

    bool runOnModule(Module &M) override;

    // StringRef getPassName() const override {
    //  return "American Fuzzy Lop Instrumentation";
    // }

  protected:
    uint32_t start_cnt = 1;
    uint32_t delta = 1;
    double sigma = 1.0;
    uint32_t share_y;
    fstream file;
    map<BasicBlock *, array<int, 2>> Params;
    map<BasicBlock *, uint32_t> Keys, SingleHash;
    map<BasicBlock *, vector<BasicBlock *>> Preds;
    vector<BasicBlock *> SingleBBS, MultiBBS, Solv, Unsolv;
    set<uint32_t> Hashes;
    map<array<uint32_t, 2>, uint32_t> HashMap;
    bool IsJoint(set<uint32_t> &A, set<uint32_t> &B)
    {
      for(auto &a : A)
        for(auto &b : B)
          if (a == b)
            return true;
      return false;
    }
    void CalcFmul()
    {
      set<uint32_t> tmpHashSet;
      uint32_t cur, edgeHash;
      bool isFind;
      for (int y = 1; y < MAP_SIZE_POW2; y++)
      {
        Hashes.clear();
        Params.clear();
        Solv.clear();
        Solv.clear();
        for(auto &BB : MultiBBS)
        // search parameters for BB
        {
          isFind = false;
          for (int x = 1; x < MAP_SIZE_POW2; x++)
          {
            for (int z = 1; z < MAP_SIZE_POW2; z++)
            {
              tmpHashSet.clear();
              cur = Keys[BB];
              // hashes for all incoming edges via Equation 2
              for(auto &p : Preds[BB])
              {
                edgeHash = ((cur >> x) ^ (Keys[p] >> y)) + z;
                // errs() << "edgeHash:\t" << edgeHash
                //        << " cur:\t" << cur
                //        << " x:\t" << x
                //        << " Keys[p]:\t" << Keys[p]
                //        << " z:\t" << z
                //        << "\n";
                tmpHashSet.insert(edgeHash);
              }
              // Found a solution for BB if no collision
              if ((tmpHashSet.size() == Preds[BB].size()) && !IsJoint(tmpHashSet, Hashes))
              {
                isFind = true;
                Params[BB] = array<int, 2>{x, z};
                share_y = y;
                Hashes.insert(tmpHashSet.begin(), tmpHashSet.end());
              }
              if (isFind)
                break;
            }
            if (isFind)
              break;
          }
          if (isFind)
            Solv.push_back(BB);
          else
            Unsolv.push_back(BB);
        }
        if (Unsolv.size() < delta || ((double)Unsolv.size() / (Unsolv.size() + Solv.size())) < sigma)
          break;
      }
    }
    void CalcFhash()
    {
      uint32_t cur;
      bool isSatisfy;
      HashMap.clear();
      for(auto &BB : Unsolv)
      {
        cur = Keys[BB];
        isSatisfy = false;
        for(auto &p : Preds[BB])
        {
          for (int i = 1; i < MAP_SIZE; i++)
            if (!Hashes.count(i))
            {
              HashMap[array<uint32_t, 2>{cur, Keys[p]}] = i;
              Hashes.insert(i);
              isSatisfy = true;
              break;
            }
          if (!isSatisfy)
          {
            errs() << "ERROR!!!\n";
            exit(1);
          }
        }
      }
    }
    void CalcFsingle()
    {
      bool isSatisfy;
      SingleHash.clear();
      for(auto &BB : SingleBBS)
      {
        isSatisfy = false;
        for (int i = 1; i < MAP_SIZE; i++)
        {
          if (!Hashes.count(i))
          {
            isSatisfy = true;
            Hashes.insert(i);
            SingleHash[BB] = i;
            break;
          }
        }
        if (!isSatisfy)
        {
          errs() << "ERROR!!!\n";
          exit(1);
        }
      }
    }
  };
}

char AFLCoverage::ID = 0;

bool AFLCoverage::runOnModule(Module &M)
{
  string file_name = M.getName().str();
  if ((file_name.rfind("/") != string::npos))
    file_name = file_name.substr(file_name.rfind("/") + 1);
  if ((file_name.find(".") != string::npos))
    file_name = file_name.substr(0, file_name.find("."));

  LLVMContext &C = M.getContext();

  IntegerType *Int8Ty = IntegerType::getInt8Ty(C);
  IntegerType *Int32Ty = IntegerType::getInt32Ty(C);

  /* Show a banner */

  char be_quiet = 0;

  if (isatty(2) && !getenv("AFL_QUIET"))
  {

    SAYF(cCYA "afl-llvm-pass " cBRI VERSION cRST " by <lszekeres@google.com>\n");
  }
  else
    be_quiet = 1;

  /* Decide instrumentation ratio */

  char *inst_ratio_str = getenv("AFL_INST_RATIO");
  uint32_t inst_ratio = 100;

  if (inst_ratio_str)
  {

    if (sscanf(inst_ratio_str, "%u", &inst_ratio) != 1 || !inst_ratio ||
        inst_ratio > 100)
      FATAL("Bad value of AFL_INST_RATIO (must be between 1 and 100)");
  }

  /* Get globals for the SHM region and the previous location. Note that
     __afl_prev_loc is thread-local. */

  GlobalVariable *AFLMapPtr =
      new GlobalVariable(M, PointerType::get(Int8Ty, 0), false,
                         GlobalValue::ExternalLinkage, 0, "__afl_area_ptr");

  GlobalVariable *AFLMyMapPtr =
      new GlobalVariable(M, PointerType::get(Int8Ty, 0), false,
                         GlobalValue::ExternalLinkage, 0, "__afl_testcase_path_ptr");

  GlobalVariable *AFLPrevLoc = new GlobalVariable(
      M, Int32Ty, false, GlobalValue::ExternalLinkage, 0, "__afl_prev_loc",
      0, GlobalVariable::GeneralDynamicTLSModel, 0, false);

  uint32_t counter = start_cnt;
  file.open("/tmp/"+file_name+"-CollAFL-mem.txt", ios::out);
  for (auto &F : M)
  {
    // errs() << "Function name:\t" << F.getName() << "\n";
    for (auto &b : F)
    {
      BasicBlock *BB = &b;
      if (BB->hasNPredecessors(0) && ((&(BB->getParent()->getEntryBlock())) != BB))
        continue;
      if (BB->hasNPredecessors(1))
        SingleBBS.push_back(BB);
      else
        MultiBBS.push_back(BB);
      for (const auto &pred : predecessors(BB))
        Preds[BB].push_back(pred);
      Keys[BB] = counter++;

      // determine the kind of instruction
      uint32_t memory_access_operation_sum = 0;
      for (BasicBlock::iterator inst = BB->begin(); inst != BB->end(); inst++)
      {
        if (inst->getOpcode() == Instruction::Alloca ||
            inst->getOpcode() == Instruction::Load ||
            inst->getOpcode() == Instruction::Store ||
            inst->getOpcode() == Instruction::GetElementPtr ||
            inst->getOpcode() == Instruction::Fence ||
            inst->getOpcode() == Instruction::AtomicCmpXchg ||
            inst->getOpcode() == Instruction::AtomicRMW)
        {
          memory_access_operation_sum++;
          // errs() << *inst << "\n";
        }
      }
      file << Keys[BB] << " " << memory_access_operation_sum << "\n";
    }
  }
  file.close();
  file.open("/tmp/"+file_name+"-CollAFL-br.txt",ios::out);
  // errs() << "SingleBBS size:\t" << SingleBBS.size() << "\n";
  // errs() << "MultiBBS size:\t" << MultiBBS.size() << "\n";
  // errs() << "Keys size:\t" << Keys.size() << "\n";
  CalcFmul();
  CalcFhash();
  CalcFsingle();
  // errs() << "After cal\n";
  // errs() << "Solv size:\t" << Solv.size() << "\n";
  // errs() << "Unsolv size:\t" << Unsolv.size() << "\n";
  // errs() << "Hashes size:\t" << Hashes.size() << "\n";
  // errs() << "Params size:\t" << Params.size() << "\n";
  // errs() << "HashMap size:\t" << HashMap.size() << "\n";
  // errs() << "SingleHash size:\t" << SingleHash.size() << "\n";
  // errs() << "share y:\t" << share_y << "\n";

  // errs() << "Param:\n";
  // for(auto &sov:Solv)
  // errs() <<Keys[sov] <<":\t"<<Params[sov][0]<<" "<<Params[sov][1]<< "\n";

  /* Instrument all the things! */

  int inst_blocks = 0;

  for(auto &F : M)
    for(auto &BB : F)
    {

      BasicBlock::iterator IP = BB.getFirstInsertionPt();
      IRBuilder<> IRB(&(*IP));

      if (AFL_R(100) >= inst_ratio)
        continue;

      /* Make up cur_loc */

      // uint32_t cur_loc = AFL_R(MAP_SIZE);
      BasicBlock *bb = &BB;
      uint32_t cur_loc = Keys[bb];

      ConstantInt *CurLoc = ConstantInt::get(Int32Ty, cur_loc);

      /* Load prev_loc */

      LoadInst *PrevLoc = IRB.CreateLoad(AFLPrevLoc);
      PrevLoc->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      Value *PrevLocCasted = IRB.CreateZExt(PrevLoc, IRB.getInt32Ty());

      /* Load SHM pointer */

      LoadInst *MapPtr = IRB.CreateLoad(AFLMapPtr);
      MapPtr->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      // Value *MapPtrIdx =
      //     IRB.CreateGEP(MapPtr, IRB.CreateXor(PrevLocCasted, CurLoc));
      Value *MapPtrIdx;
      if(Params.count(bb))//InstrumentFmul
      {
        array<int ,2> arr=Params[bb];
        CurLoc = ConstantInt::get(Int32Ty, cur_loc>>arr[0]);
        Value *h = IRB.CreateXor(PrevLocCasted, CurLoc);
        MapPtrIdx = IRB.CreateGEP(MapPtr, IRB.CreateAdd(h, ConstantInt::get(Int32Ty, arr[1])));
      }
      else if (SingleHash.count(bb))//InstrumentFsingle
      {
        MapPtrIdx = IRB.CreateGEP(MapPtr, ConstantInt::get(Int32Ty, SingleHash[bb]));
      }
      else
        continue;
      

      /* Update bitmap */

      LoadInst *Counter = IRB.CreateLoad(MapPtrIdx);
      Counter->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      Value *Incr = IRB.CreateAdd(Counter, ConstantInt::get(Int8Ty, 1));
      IRB.CreateStore(Incr, MapPtrIdx)
          ->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));

      /* Set prev_loc to cur_loc >> 1 */

      // StoreInst *Store =
      //     IRB.CreateStore(ConstantInt::get(Int32Ty, cur_loc >> 1), AFLPrevLoc);
      StoreInst *Store =
          IRB.CreateStore(ConstantInt::get(Int32Ty, cur_loc >> share_y), AFLPrevLoc);
      Store->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));

      inst_blocks++;

      LoadInst *MapMyPtr = IRB.CreateLoad(AFLMyMapPtr);
      MapMyPtr->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
      Value *MapMyPtrIdx = IRB.CreateGEP(MapMyPtr, ConstantInt::get(Int32Ty, cur_loc));
      IRB.CreateStore(ConstantInt::get(Int8Ty, 1), MapMyPtrIdx)
          ->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
    }

    for(auto &bb : Unsolv)
    {

      if (AFL_R(100) >= inst_ratio)
        continue;

      /* Make up cur_loc */

      uint32_t cur_loc = Keys[bb];
      for(auto &prevbb : Preds[bb])
      {
        uint32_t pre_loc = Keys[prevbb];
        if (HashMap.count(array<uint32_t, 2>{cur_loc, pre_loc}))
        {
          BasicBlock *dummyBB = SplitEdge(prevbb, bb);
          Keys[dummyBB] = counter++;
          HashMap[array<uint32_t,2>({Keys[dummyBB], pre_loc})]=HashMap[array<uint32_t,2>({cur_loc, pre_loc})];
          // for (int i = 1; i < MAP_SIZE; i++)
          // {
          //   if (!Hashes.count(i))
          //   {
          //     Hashes.insert(i);
          //     SingleHash[dummyBB] = i;
          //     break;
          //   }
          // }
          BasicBlock::iterator IP = dummyBB->getFirstInsertionPt();
          IRBuilder<> IRB(&(*IP));
          ConstantInt *CurLoc = ConstantInt::get(Int32Ty, cur_loc);

          /* Load prev_loc */

          LoadInst *PrevLoc = IRB.CreateLoad(AFLPrevLoc);
          PrevLoc->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
          Value *PrevLocCasted = IRB.CreateZExt(PrevLoc, IRB.getInt32Ty());

          /* Load SHM pointer */

          LoadInst *MapPtr = IRB.CreateLoad(AFLMapPtr);
          MapPtr->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
          Value *MapPtrIdx = IRB.CreateGEP(MapPtr, ConstantInt::get(Int32Ty, HashMap[array<uint32_t,2>({cur_loc, pre_loc})]));

          /* Update bitmap */

          LoadInst *Counter = IRB.CreateLoad(MapPtrIdx);
          Counter->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
          Value *Incr = IRB.CreateAdd(Counter, ConstantInt::get(Int8Ty, 1));
          IRB.CreateStore(Incr, MapPtrIdx)
              ->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));

          /* Set prev_loc to cur_loc >> 1 */

          // StoreInst *Store =
          //     IRB.CreateStore(ConstantInt::get(Int32Ty, cur_loc >> 1), AFLPrevLoc);
          StoreInst *Store =
              IRB.CreateStore(ConstantInt::get(Int32Ty, cur_loc >> share_y), AFLPrevLoc);
          Store->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));

          inst_blocks++;

          LoadInst *MapMyPtr = IRB.CreateLoad(AFLMyMapPtr);
          MapMyPtr->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
          Value *MapMyPtrIdx = IRB.CreateGEP(MapMyPtr, ConstantInt::get(Int32Ty, Keys[dummyBB]));
          IRB.CreateStore(ConstantInt::get(Int8Ty, 1), MapMyPtrIdx)
              ->setMetadata(M.getMDKindID("nosanitize"), MDNode::get(C, None));
        }
      }
    }

  /* Say something nice. */

  if (!be_quiet)
  {

    if (!inst_blocks)
      WARNF("No instrumentation targets found.");
    else
      OKF("Instrumented %u locations (%s mode, ratio %u%%).",
          inst_blocks, getenv("AFL_HARDEN") ? "hardened" : ((getenv("AFL_USE_ASAN") || getenv("AFL_USE_MSAN")) ? "ASAN/MSAN" : "non-hardened"), inst_ratio);
  }

  // calculate edge and its hash , then write them down
  for (auto &F : M)
  {
    // errs() << F.getName() << "\n";
    for (auto &BB : F)
    {
      BasicBlock *preBB = &BB;
      for (succ_iterator SI = succ_begin(&BB), SE = succ_end(&BB); SI != SE; SI++)
      {
        BasicBlock *curBB = *SI;
        uint32_t pre = Keys[preBB];
        uint32_t cur = Keys[curBB];
        uint32_t hashRes = -1;
        // uint32_t origin = -1;

        if (Params.count(curBB))
        {
          array<int, 2> arr = Params[curBB];
          hashRes = ((cur >> arr[0]) ^ (pre >> share_y)) + arr[1];
          // origin = 1;
        }
        else if (HashMap.count(array<uint32_t, 2>{cur, pre}))
        {
          hashRes = HashMap[array<uint32_t, 2>{cur, pre}];
          // origin = 2;
        }
        else if (SingleHash.count(curBB))
        {
          hashRes = SingleHash[curBB];
          // origin = 3;
        }
        else
        {
          // errs() << pre << " -> " << cur << "\tno hash"
          //        << "\n";
          continue;
        }
        // errs() << pre << " -> " << cur << "\thash:" << hashRes;
        // if (origin == 1)
        //   errs() << "\tfrom Mul Solv\n";
        // else if (origin == 2)
        //   errs() << "\tfrom Mul UnSolv\n";
        // else if (origin == 3)
        //   errs() << "\tfrom Single\n";
        file << pre << " " << cur << " " << hashRes << "\n";
      }
    }
  }
  file.close();

  return true;
}

static void registerAFLPass(const PassManagerBuilder &,
                            legacy::PassManagerBase &PM)
{

  PM.add(new AFLCoverage());
}

static RegisterStandardPasses RegisterAFLPass(
    PassManagerBuilder::EP_ModuleOptimizerEarly, registerAFLPass);

static RegisterStandardPasses RegisterAFLPass0(
    PassManagerBuilder::EP_EnabledOnOptLevel0, registerAFLPass);