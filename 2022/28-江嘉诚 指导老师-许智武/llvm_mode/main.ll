; ModuleID = 'main.c'
source_filename = "main.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__afl_area_ptr = external global i8*
@__afl_testcase_path_ptr = external global i8*
@__afl_prev_loc = external thread_local global i32

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fun1() #0 {
  %1 = load i32, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %2 = load i8*, i8** @__afl_area_ptr, align 8, !nosanitize !2
  %3 = xor i32 %1, 0
  %4 = add i32 %3, 1
  %5 = getelementptr i8, i8* %2, i32 %4
  %6 = load i8, i8* %5, align 1, !nosanitize !2
  %7 = add i8 %6, 1
  store i8 %7, i8* %5, align 1, !nosanitize !2
  store i32 0, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %8 = load i8*, i8** @__afl_testcase_path_ptr, align 8, !nosanitize !2
  %9 = getelementptr i8, i8* %8, i32 1
  store i8 1, i8* %9, align 1, !nosanitize !2
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fun2() #0 {
  %1 = load i32, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %2 = load i8*, i8** @__afl_area_ptr, align 8, !nosanitize !2
  %3 = xor i32 %1, 1
  %4 = add i32 %3, 1
  %5 = getelementptr i8, i8* %2, i32 %4
  %6 = load i8, i8* %5, align 1, !nosanitize !2
  %7 = add i8 %6, 1
  store i8 %7, i8* %5, align 1, !nosanitize !2
  store i32 1, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %8 = load i8*, i8** @__afl_testcase_path_ptr, align 8, !nosanitize !2
  %9 = getelementptr i8, i8* %8, i32 2
  store i8 1, i8* %9, align 1, !nosanitize !2
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = load i32, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %2 = load i8*, i8** @__afl_area_ptr, align 8, !nosanitize !2
  %3 = xor i32 %1, 1
  %4 = add i32 %3, 1
  %5 = getelementptr i8, i8* %2, i32 %4
  %6 = load i8, i8* %5, align 1, !nosanitize !2
  %7 = add i8 %6, 1
  store i8 %7, i8* %5, align 1, !nosanitize !2
  store i32 1, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %8 = load i8*, i8** @__afl_testcase_path_ptr, align 8, !nosanitize !2
  %9 = getelementptr i8, i8* %8, i32 3
  store i8 1, i8* %9, align 1, !nosanitize !2
  %10 = alloca i32, align 4
  %11 = alloca [20 x i8], align 16
  store i32 0, i32* %10, align 4
  %12 = getelementptr inbounds [20 x i8], [20 x i8]* %11, i64 0, i64 0
  %13 = call i32 (i8*, ...) bitcast (i32 (...)* @gets to i32 (i8*, ...)*)(i8* %12)
  br label %14

14:                                               ; preds = %0, %44
  %15 = load i32, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %16 = load i8*, i8** @__afl_area_ptr, align 8, !nosanitize !2
  %17 = xor i32 %15, 2
  %18 = add i32 %17, 1
  %19 = getelementptr i8, i8* %16, i32 %18
  %20 = load i8, i8* %19, align 1, !nosanitize !2
  %21 = add i8 %20, 1
  store i8 %21, i8* %19, align 1, !nosanitize !2
  store i32 2, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %22 = load i8*, i8** @__afl_testcase_path_ptr, align 8, !nosanitize !2
  %23 = getelementptr i8, i8* %22, i32 4
  store i8 1, i8* %23, align 1, !nosanitize !2
  %24 = getelementptr inbounds [20 x i8], [20 x i8]* %11, i64 0, i64 0
  %25 = load i8, i8* %24, align 16
  %26 = sext i8 %25 to i32
  %27 = icmp ne i32 %26, 0
  br i1 %27, label %28, label %36

28:                                               ; preds = %14
  %29 = load i32, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %30 = load i8*, i8** @__afl_area_ptr, align 8, !nosanitize !2
  %31 = getelementptr i8, i8* %30, i32 1
  %32 = load i8, i8* %31, align 1, !nosanitize !2
  %33 = add i8 %32, 1
  store i8 %33, i8* %31, align 1, !nosanitize !2
  store i32 2, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %34 = load i8*, i8** @__afl_testcase_path_ptr, align 8, !nosanitize !2
  %35 = getelementptr i8, i8* %34, i32 5
  store i8 1, i8* %35, align 1, !nosanitize !2
  call void @fun1()
  br label %52

36:                                               ; preds = %14
  %37 = load i32, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %38 = load i8*, i8** @__afl_area_ptr, align 8, !nosanitize !2
  %39 = getelementptr i8, i8* %38, i32 3
  %40 = load i8, i8* %39, align 1, !nosanitize !2
  %41 = add i8 %40, 1
  store i8 %41, i8* %39, align 1, !nosanitize !2
  store i32 3, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %42 = load i8*, i8** @__afl_testcase_path_ptr, align 8, !nosanitize !2
  %43 = getelementptr i8, i8* %42, i32 6
  store i8 1, i8* %43, align 1, !nosanitize !2
  call void @fun2()
  br label %44

44:                                               ; preds = %36
  %45 = load i32, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %46 = load i8*, i8** @__afl_area_ptr, align 8, !nosanitize !2
  %47 = getelementptr i8, i8* %46, i32 5
  %48 = load i8, i8* %47, align 1, !nosanitize !2
  %49 = add i8 %48, 1
  store i8 %49, i8* %47, align 1, !nosanitize !2
  store i32 3, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %50 = load i8*, i8** @__afl_testcase_path_ptr, align 8, !nosanitize !2
  %51 = getelementptr i8, i8* %50, i32 7
  store i8 1, i8* %51, align 1, !nosanitize !2
  br label %14

52:                                               ; preds = %28
  %53 = load i32, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %54 = load i8*, i8** @__afl_area_ptr, align 8, !nosanitize !2
  %55 = getelementptr i8, i8* %54, i32 6
  %56 = load i8, i8* %55, align 1, !nosanitize !2
  %57 = add i8 %56, 1
  store i8 %57, i8* %55, align 1, !nosanitize !2
  store i32 4, i32* @__afl_prev_loc, align 4, !nosanitize !2
  %58 = load i8*, i8** @__afl_testcase_path_ptr, align 8, !nosanitize !2
  %59 = getelementptr i8, i8* %58, i32 8
  store i8 1, i8* %59, align 1, !nosanitize !2
  %60 = load i32, i32* %10, align 4
  ret i32 %60
}

declare dso_local i32 @gets(...) #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 0160ad802e899c2922bc9b29564080c22eb0908c)"}
!2 = !{}
