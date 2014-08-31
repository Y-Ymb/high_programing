################################
# Makefile for Benkyokai2014
# by Makoto Shimazu <shimazu@eidos.ic.i.u-tokyo.ac.jp>
################################

# 自分のつくったものに合うように変更すること
YOURSRCS  := mymulmat1.cpp
YOURCLASS := MyMulMat
YOURCLASSHEADER := mymulmat.h

# FX10/XEONPHI/MYLOCAL
PLATFORM := MYLOCAL
# MPIを利用するときには1,そうでないときは0にする
USEMPI   := 1

# 適宜変更しても大丈夫
ifeq ($(PLATFORM),FX10)
CXX       = FCCpx
LD        = FCCpx
CXXFLAGS += -Xg 
LDFLAGS  += 
LIBS     += 
endif
ifeq ($(PLATFORM),XEONPHI)
CXX       = icc
LD        = icc
CXXFLAGS += -MMD -W -Wall -mmic
LDFLAGS  += -mmic
LIBS     +=
endif
ifeq ($(PLATFORM),MYLOCAL)
CXX       = mpic++
LD        = mpic++
CXXFLAGS += -MMD -W -Wall -fopenmp
LDFLAGS  += -fopenmp
LIBS     +=
endif

# 測定用
VERSION   := 0.2.2
MAIN      := main.bin
SRCDIR	  := src
OBJDIR	  := obj
# LIBS      += 
MDIR	  := measure
MSRCS	  := main.cpp tester.cpp dataset.cpp
MOBJS	  := $(MSRCS:.cpp=.o)
MOBJS_FULL = $(addprefix $(OBJDIR)/$(MDIR)/, $(MOBJS))
MDEPS     := $(MSRCS:.cpp=.d)
MDEPS_FULL = $(addprefix $(OBJDIR)/$(MDIR)/, $(MDEPS))
# Path to your codes
YOUROBJS      := $(YOURSRCS:.cpp=.o)
YOUROBJS_FULL  = $(addprefix $(OBJDIR)/, $(YOUROBJS))
YOURDEPS      := $(YOURSRCS:.cpp=.d)
YOURDEPS_FULL  = $(addprefix $(OBJDIR)/, $(YOURDEPS))

.PHONY: all
all: build

.PHONY: build
build: dirs $(MAIN)

$(MAIN): $(MOBJS_FULL) $(YOUROBJS_FULL)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -DVERSION=\"$(VERSION)\" -DMMCLASS=$(YOURCLASS) -DUSEMPI=$(USEMPI) -include $(SRCDIR)/$(YOURCLASSHEADER) -c -o $@ $< $(LIBS)


.PHONY: dirs
dirs: $(OBJDIR) $(OBJDIR)/$(MDIR)

$(OBJDIR):
	mkdir -pv $@
$(OBJDIR)/$(MDIR):
	mkdir -pv $@

.PHONY: clean
clean:
	rm -rf $(MAIN) $(OBJDIR)


-include $(YOURDEPS_FULL) $(MDEPS_FULL)
