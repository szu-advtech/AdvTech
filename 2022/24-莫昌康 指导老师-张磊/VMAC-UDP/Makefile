CONFIG_MODULE_SIG = n
KCOV_INSTRUMENT := n

# 如果定义了KERNELRELEASE，那么我们已经从内核构建系统中被调用，并且可以使用它的语言。
ifneq ($(KERNELRELEASE),)
	obj-m += vmac_udp.o
	vmac_udp-y := \
		main.o 
# 否则，我们直接从命令行调用；调用内核构建系统。
else
	KERNELDIR ?= /lib/modules/$(shell uname -r)/build
	PWD := $(shell pwd)
default:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
endif

# obj-m += vmac.o
# vmac-y := \
#         main.o \
#         clean.o \
#         queue.o \
#         dack.o \
#         rx.o  \
#         tx.o  \
#         hooks.o \
#         rate.o