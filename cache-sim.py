import sys, argparse, math
import numpy as np
import typing, random 
import timeit
from tqdm import tqdm

#import Cache as Cache

sys.setrecursionlimit(1500)

NUMSETS = 0  
NUMBLOCKS = 0 
bit_index = 0 
bit_offset = 0 
class DataBlock: 
    """
    Note: one double = one word = 8 bytes
    :attribute self.size: the size of the block in bytes, default 64 bytes so in 
    the default, there are 8 words in one block. 
    :type self.size: int

    :attribute self.b_data: An array of doubles of self.size 
    :type self.b_data: numpy array of floats 
    """
    size = 64
    

    def __init__(self):
        # default size of the data block is 64 bytes 
        #8 words * 8 bytes = 64 bytes in one data block 
        #addresses can't be set when the datablock is intialized, must be 
        #updated when the RAM is created
        self.b_data = np.arange(0, self.size, 8, dtype=float) 
        # just doing strides of 8 so we have the correct number of elements in the array 
    
    def __iter__(self):
        for w in self.b_data:
            yield w
        return

class Address:    
    """
    :attribute self.address: the address of the given block of 64 bits 
    :type self.address: int # in python this will just be 64 bits

    """

    #right shift the address by the number of offset and index bits to get the tag 
    #tag = #np.uint64(self.address) >> np.uint64((offset+index)) 
    def __init__(self, address: int = 0) -> None:
        self.address = address

    #gets the double offset of a DataBlock 
    def get_offset(self) -> int:
        block_size_bytes = DataBlock.size
        offset = self.address % block_size_bytes # get the byte offset 
        return int(offset) // 8 # can int divide by 8 again to get the double/word offset

    """
    return the index of an address 

    for direct mapped: return the block address % number of blocks in cache (only one place it can go)
    for set associative: return the block address % the number of sets (only one set it can go )
    """
    def get_index(self) -> int:
        block_address = self.address // DataBlock.size
        if NUMSETS == 1:
            return block_address % Cache.numBlocks
        else:
            index = block_address % NUMSETS
            return index

    """
    return the tag bits of an address in the form of an integer 

    process: find the number of index and offset bits of the address and right 
    shift the address (index + offset) to isolate just the tag 

    """
    def get_tag(self) -> int: 
        
        return ((self.address) >> (bit_index + bit_offset)) & 0xFFFFF
    
         



    
# class RamBlock:
#     """Ram Blocks make up the blocks that fit in RAM
#     self.ram_address is the address of the block (multiple of datablock size)
#     self.ram_data is the DataBlock associated with the address 
#     """
#     def __init__(self, address: Address, data: DataBlock) -> None: 
#         self.ram_address = address
#         self.ram_data = data


class Ram: 
    """
    :attribute self.numBlocks: the number of blocks in ram 
    :type self.numBlocks: float 

    :attribute self.data: the data structure containing all the blocks in Ram
    note that the self.data contains elements of RamBlock
    :type self.data: an array of DataBlocks of size numBlocks 
    """
    #need to figure out the default number of blocks in the ram 
    #careful with defaults here given changeable flags/parameters 
    
    def __init__(self, numBlocks = 64):
        self.numBlocks = numBlocks 
        #ram_blocks = []
        block_size = DataBlock.size
        L = np.empty(self.numBlocks, dtype = object)
        for N in range(0, self.numBlocks):
            #intialize addresses sequentially 
            #new_block = (Address(N * block_size), DataBlock())
            #ram_blocks.append(new_block)
            L[N] = (Address(N * block_size), DataBlock())
        
        self.data = L

    #get the RamBlock where address points to 
    def getBlock(self, address: Address) -> tuple:
        #print(address)
        address = address.address 
        block_num = address // DataBlock.size
        return self.data[block_num]
    
    def setBlock(self, address: Address, value: DataBlock) -> None:
        address = address.address
        block_size = DataBlock.size
        block_num = address // block_size
        self.data[block_num] = (Address(block_num * block_size), value) 
        #note that set block in ram always has memory blocks as the size of a datablock
        #so for a 64 byte block, setblock will always set as a multiple of 64 

class CacheBlock:
    """ Cache Block are the blocks that make up data in cache 
    self.valid is a 1 or a 0 depending on whether or not there is an address/datablock 
        in the cache location (everything is intialized as 0)
    
    self.address = full address of where the block is located in memory 
    
    self.datablock = the datablock associated with the address  """
    def __init__(self, valid: int, address: Address, datablock: DataBlock):
        self.valid = valid 
        self.address = address
        self.datablock = datablock

    def is_valid(self):
        if self.valid == 1:
            return True
        if self.valid == 0:
            return False 

    def print_block_data(self): 
        print(f"Full Block: {self}")
        print(f"Valid: {self.valid}")
        if (self.address):
            print(f"Address: {self.address.address}")
        if (self.datablock):
            print(f"DataBlock: {self.datablock}")
        return ""



    

class Cache: 
    """
    instance attributes
    :attribute self.numSets: the number of sets in the cache
    :type self.numSets: int
    
    :attribute self.ram: pass in the ram instance 

    :attribute self.blocks: DataBlock[numSets][NumBlocks]

    self.readhit
    self.readmiss
    self.writehit
    self.writemiss

    class attributes: 
    :attribute self.numBlocks: the number of blocks in cache 
    :type self.numBlocks: int 

    associativity: the level of set associativity the cache has 
    1 = direct mapped 
    n = numBlocks = fully associative 
    """
    #Needs to be calculated based on the parameters 
    numBlocks = 4 #cache size // block size = # of blocks % num Sets = blocks per set 
    associativity = 1
    if associativity == 1:
        numSets = 1
    elif associativity == numBlocks:
        numSets = numBlocks
    else:
        numSets = numBlocks // associativity 
    #careful with defaults here given changes in flags/parameters (numSets needs to be variable)

    def __init__(self, ram: Ram, replace: str) -> None:
        self.ram = ram
        self.replace = replace
        #this stores cache tuples 
        self.LRU = [[] for i in range(0, self.numSets)]

        blocks_per_set = self.numBlocks // self.numSets 
        #print(f"number of sets: {self.numSets}") #number of sets doesn't update the class when associativity updates
        #print(f"numblocks: {self.numBlocks}")
        L = np.empty(shape=(self.numSets, blocks_per_set), dtype = object)
        for i in range(0, self.numSets):
            for j in range(0, blocks_per_set):
                tmp = (0, None, None)
                L[i][j] = tmp
        self.blocks = L
        self.ReadHit = 0 
        self.ReadMiss = 0 
        self.WriteHit = 0 
        self.WriteMiss = 0 




    """
    returns a CacheBlock or a RamBlock
    
    if direct mapped:
        go to the index location in the cache 
            if there is already a block there 
                check if it matches the tag of the passed in address
                    if so return the CacheBlock (read hit in get Double)
                if it doesn't match the tag of the passed in address
                    get the block from memory 
                    place it in the cache
                    and return the RamBlock (note read miss will be calculated in getDouble)
            if there isn't already a block there
                get the corresponding block from memory 
                bring it into the cache
                return the RamBlock (read miss in getDouble)

    if full associativie
    if not direct mapped: 
        go to the desired set based off of the index 
            for every block in the set 
                check if a block is valid 
                    if it is valid
                        check if it matches the tag of the passed address
                        if it matches the tag 
                            return the CacheBlock (read hit)
                    if it isn't valid
                        just continue 
            if no block in the set returns, then it must not be in cache
                so go to memory to get the RamBlock
                set the block in cache 
                return RamBlock (read miss) 
    """
    def getBlock(self, address: Address):
        tag = address.get_tag()
        index = address.get_index()
        my_block = None
        if self.associativity == 1: 
            #print(index)
            my_block = self.blocks[0][index] 
            #print(my_block)
            #print("Get Block: my Block")
            #print(my_block)
            #if my_block[0] == 1 : # if the position is already valid, there it must be some CacheBlock there
                #block_tag = my_block[1]
            if my_block[1] == tag: #check if it is the block that I want
                return my_block # If it is, return the CacheBlock 
            else: 
                new_block = cache.ram.getBlock(address)
                self.setBlock(new_block[0], new_block[1])
                return new_block
            # else: #if I go to the index and the block is not valid, this is a read miss, pull the block into cache 
            #     new_block = ram.getBlock(address) # note this does not return a cache block, it returns a RamBlock
            #     self.setBlock(new_block[0], new_block[1]) 
            #     # bring the new block into cache 
            #     # use the ram address and data, ram address will always be multiple of block size
            #     return new_block # this returns RamBlock
        elif self.associativity == self.numBlocks:

            #need to implement LRU+ FIFO here 
            #print(index)
            my_block = self.blocks[0][index]
            #print(my_block)
            # print(address.address)
            # my_block = self.blocks[index][0]
            # print(my_block.is_valid())
            if my_block[0] == 1: # if the position is already valid, there it must be some CacheBlock there
                block_tag = my_block[1]
                if block_tag == tag: #check if it is the block that I want
                    return my_block # If it is, return the CacheBlock 
            else: 
                new_block = cache.ram.getBlock(address)
                self.setBlock(new_block[0], new_block[1])
                return new_block
        else:
            #need to update this section based on above
            #if there is a read miss you need to bring the block into cache 
            #print(index)
            set = self.blocks[index]
            for block in set:
                #for low level addresses, all the tags will be the same --> seems to be resolved?
                #that's because all the addresses are the same right now 
                if block[0] == 1:
                    block_tag = block[1]
                    if block_tag == tag:
                        if self.replace == "LRU":
                            #if you read the block and its already in the LRU queue, move it to the end
                            # if you read a block and its not already in the LRU queue, add it to the end 
                            #remove LRU from the first item in the list
                            if block in self.LRU[index]:
                                self.LRU[index].remove(block)
                                self.LRU[index].append(block)
                                # self.LRU[index].append(my_block)
                            else:
                                self.LRU[index].append(block)
                        return block #this is a CacheBlock! 
                else:
                    continue
            new_block = ram.getBlock(address)
            self.setBlock(new_block[0], new_block[1])
            if self.replace == "LRU":
                block = (1, new_block[0].get_tag(), new_block[1])
                self.LRU[index].append(block)
            return new_block # return a RamBlock 
    

    """
    set a CacheBlock in cache given an address and a Datablock 
    
    3 different policies: random, LRU, FIFO. Right now just doing random replacement policy for 
    set associativity 

    for DM, must auto evict 

    if a direct mapped cache: 
        find the block you want to change in cache by going to its index 
        if the block is valid 
            set the address to the block address I want to change? --> is this necessary? left out for now 
            consider write hits or not? 
        update the block in cache with the address and value (write hit)
    if it is a fully associative cache 
        find the block you want to change in cache by going to its index
            need to be based on LRU, FIFO, random --> not implemented fully 
        update the block in cache with the passed in address and value 
    if it is a set associative cache 
        find the index of the set you want to place the block in 
        randomly pick a block to replace (or LRU or FIFO) -> not implemented fully 
        replace with a new cache block 
    """
    def setBlock(self, address: Address, value: DataBlock) ->  None: 
        #print(f"Initialized Address in Set Block: {address}")
        index = address.get_index()
        tag = address.get_tag()

        if self.associativity == 1: 
            #my_block = self.blocks[0][index]
            #if my_block.is_valid():
            #    address = my_block.address
            self.blocks[0][index] = (1, tag, value) 
        elif self.associativity == self.numBlocks:
            #my_block = self.blocks[index][0]
            #if my_block.is_valid():
            #    address = my_block.address  
            self.blocks[0][index] = (1, tag, value)
        else: 
            #need to keep track of cache replacement policy 
            #assuming random placement, need valid bit --> valid bit added
            #what about LRU and FIFO? 
            #print("in Set Block associativity logic")
            blocks_per_set = self.numBlocks // self.numSets
            if self.replace == "LRU":
                #basically I want to place in spot that has 0 first before I randomly replace
                #So, if I need to replace a block
                #I should find the LRU (the first item in the list) and remove it in the cache 
                #I should replace the block I want in the position the removed block was in 
                #print(self.LRU[0])
                #print(index)
                #and after I replace the block with the new block, I want to add teh new block to LRU
                for i in range(0, blocks_per_set):
                    if self.blocks[index][i][0] == 0: 
                        new_block = (1, tag, value)
                        self.blocks[index][i] = new_block 
                        self.LRU[index].append(new_block)
                        return
                
                try:
                    #just removing everything even if there isn't a full set 
                    #print(len(self.LRU[index]))
                    #print(blocks_per_set)
                    if len(self.LRU[index]) == blocks_per_set:
                        #print("pop")
                        remove_block = self.LRU[index].pop(0)
                    else:
                        remove_block = None
                except:
                    remove_block = None
                new_block = (1, tag, value)
                for i in range(0, blocks_per_set):
                    if self.blocks[index][i] == remove_block:
                        #print("found block to remove") 
                        #print(f"removed block: {self.blocks[index][i]}")
                        self.blocks[index][i] = new_block
                        #print(f"new block: {self.blocks[index][i]}")
                self.LRU[index].append(new_block)
                #print(f"index {index}")
                #print(f"LRU list {self.LRU[index]}")
                #print(new_block)

            if self.replace == "random":
                #basically I want to place in spot that has 0 first before I randomly replace
                #i think this is a bad idea because it basically intializes everything to the first block when it's initialized
                #creates more conflict misses 
                for i in range(0, blocks_per_set):
                    if self.blocks[index][i][0] == 0: 
                        self.blocks[index][i] = (1, tag, value) 
                  
                replace = random.randint(0, self.associativity-1)
                self.blocks[index][replace] = (1, tag, value) 

            #my_block = self.blocks[index][replace]
            #if my_block.is_valid():
            #    address = my_block.address
            

    """
    get the double at the desired address (an int) 

    logic:
    call get block on the address which will either return a Cacheblock or a Ram Block
    note the address is passed in as an integer and turned into an address 
    if it is a CacheBlock
        get the block offset from the address
        get the double from the CacheBlock
        increment the read hit bc. the block came from cache
        return the the doudble 
    if it is a RamBlock 
        get the block offset from the address
        get the double from the Ramblock
        increment the read miss bc. the block came from RAM 
        (the block is pulled into Cache from main mem in GetBlock)
        return the double 

    """
    def getDouble(self, address: int) -> float:
        address = Address(address) # need to be careful with the address thingys 
        my_block = self.getBlock(address) # either a CacheBlock or a RamBlock
        #print("Get Double")
        #print(my_block)
        if not isinstance(my_block[0], Address): #for cache blocks, position 0 of the tuple should be an int of the valid bit 
            #my_block.print_block_data()
            #print("Get Double Cache Block")
            offset = address.get_offset()
            #if my_block.is_valid(): #the Cache Block is in memory
            my_double = my_block[2].b_data[offset] #update the cache blocks data
            #else: #it's a cold cache 
            cache.ReadHit +=1 
            return my_double
        # this isn't guaranteed, need to be careful depending on whether it is a CacheBlock or a Data Block 
        else: #if its not a cache block, then it has to be a RamBlock
            #print("Get Double RAM Block")
            #print(f"Read miss Block Address: {address.address}")
            offset = address.get_offset()
            #print(f"Block address: {address.address}")
            #print(my_block.ram_data)
            my_double = my_block[1].b_data[offset] 
            cache.ReadMiss +=1 
            return my_double 

    """
    set the double in a block 
    set the block in cache
    update the block in ram if need be 

    logic:
    call the cache getblock function on the address 
    returns a RamBlock or CacheBlock 

    if it is a CacheBlock
        get the block offset 
        update the datablock in CacheBlock with the value (datablock and cache is updated)
        update the block in RAM (write through)
        increment Write hit (write hit because the block you want to write to is in cache)
    if it is a RAMBlock 
        get the block offset 
        update the block in Ram with the correct double (write through)
        bring the updated block into cache (write allocate)
        increment write miss (write miss because the block you want is not already in cache)
    """
    def setDouble(self, address: int, value) -> None:
        address = Address(address) # need to be careful with the address objects 
        my_block = self.getBlock(address) 
        #need a way to determine if the block was gotten from RAM or from memory 
        #can determine whether my_block is a CacheBlock or a (Address, DataBlock)
        #can use is instance to determine typing 
        # and then once the type is figured out, need to keep or turn into a cache block 
        # and then update the cache 
        if not isinstance(my_block[0], Address): #this doesn't work entirely because need to find the datablock
            #write hit 
            #print("Set Double: Cache Block")
            #print(my_block)
            offset = address.get_offset()
            #print(my_block.datablock)
            my_block[2].b_data[offset] = value #block is updated and updated in cache
            self.ram.setBlock(address, my_block[2]) #write through
            cache.WriteHit +=1 
        else: #if it isn't a cache block, it must be a RamBlock
            #print("Set Double: Ram Block")
            offset = address.get_offset()
            my_block[1].b_data[offset] = value #block data is updated
            self.setBlock(my_block[0], my_block[1]) # bring the updated block into cache
            cache.WriteMiss +=1 

            
class CPU: 
    def __init__(self, cache: Cache) -> None:
        self.cache = cache
        self.instruction_count = 0
        # doesn't have any attributes 

    def loadDouble(self, address: Address) -> float: 
        self.instruction_count+=1 
        return self.cache.getDouble(address)
    
    def storeDouble(self, address: Address, value: float) -> None: 
        self.instruction_count+=1
        self.cache.setDouble(address, value)
    
    def addDouble(self, value1: float, value2: float) -> float: 
        self.instruction_count+=1
        return value1 + value2 
    
    def multDouble(self, value1: float, value2: float) -> float: 
        self.instruction_count+=1
        return value1 * value2 


            
            


        


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default = 65536) #size of cache, equivalent to 1024 blocks of 64 bytes
    parser.add_argument('-b', type=int, default = 64) #size of data block in bytes
    parser.add_argument('-n', type=int, default = 2) #n-way set associativity, note default should be 2 
    parser.add_argument('-r', type=str, default = "LRU") #replacement policy
    parser.add_argument('-a', type=str, default = "mxm_block") # the algorithm
    parser.add_argument('-d', type=int, default = 480) #the vector or matrix dimensions 
    #parser.add_argument('-p', type=int) enables the printing of the solution of matrix or daxpy 
    parser.add_argument('-f', type=int, default = 32)

    #parser.add_argument('-r', type=str) #replacement policy: random, LRU, FIFO
    args = parser.parse_args()
    

    input_block_size = args.b
    DataBlock.size = input_block_size

    input_n = args.n 
    Cache.associativity = input_n

    input_c = args.c 
    Cache.numBlocks = input_c // DataBlock.size 
    if Cache.associativity > 1: 
        Cache.numSets = Cache.numBlocks // Cache.associativity
    elif Cache.associativity == Cache.numBlocks:
        Cache.numSets = Cache.associativity
    else:
        Cache.numSets = 1 

    NUMSETS = Cache.numSets
    NUMBLOCKS = Cache.numBlocks
    bit_offset = int(math.log2(DataBlock.size))
    bit_num_sets = NUMSETS
    if bit_num_sets == 1:
        #if its direct mapped, number of index bits should be log2 of the number of cache blocks 
        bit_index = int(math.log2(NUMBLOCKS))
    else:
        #if its set associative, the number of index bits should be log2 of the number of sets 
        bit_index = int(math.log2(bit_num_sets))
    input_a = args.a
    input_r = args.r
    input_d = args.d
    input_f = args.f

    ram_num_blocks = (2**22)//input_block_size
    ram = Ram(ram_num_blocks)
    cache = Cache(ram, input_r)
    #print(cache.LRU)
    myCpu = CPU(cache)

    #cache of 4 blocks 
    #two way set associativity 
    # test_blocks = [] 
    # for i in range(0, 8):
    #     test_block = DataBlock()
    #     test_block.b_data = np.arange(0, 64, 8, dtype=float)
    #     test_blocks.append(test_block)
    # cache.setBlock(Address(0), test_blocks[0])
    # cache.setBlock(Address(64), test_blocks[1])
    # cache.setBlock(Address(128), test_blocks[2])
    # cache.setBlock(Address(192), test_blocks[3])
    # print(test_blocks)
    # print(cache.blocks)
    # cache.setBlock(Address(256), test_blocks[4])
    # cache.setBlock(Address(384), test_blocks[6])
    # print(cache.blocks)

    if input_a == 'daxpy':
        sz = 8
        n = args.d
        a = list(range(0, n*sz, sz))
        b = list(range(n*sz, 2*n*sz, sz))
        c = list(range(2*n*sz, 3*n*sz, sz))

        # for i in range(n):
        #     myCpu.storeDouble(address=a[i], value = i)
        #     myCpu.storeDouble(address=b[i], value=2*i)
        #     myCpu.storeDouble(address=c[i], value=0)
        
        register0 = 3
        for i in range(n):
            register1 = myCpu.loadDouble(a[i])
            register2 = myCpu.multDouble(register0, register1)
            register3 = myCpu.loadDouble(b[i])
            register4 = myCpu.addDouble(register2, register3)
            myCpu.storeDouble(c[i], register4)

        

    # if input_a == "mxm_block":
    #     blocking_factor = input_f
    #     sz = 8 
    #     n = args.d
    #     x = np.ones((n, n))
    #     y = np.ones((n, n))
    #     z = np.ones((n, n))
    #     #initialize some addresses for 
    #     for i in range(0, n):
    #         for j in range(0,n):
    #             y[i][j] = (j*sz) + ((i*n) * sz)
    #             z[i][j] = (j*sz) + ((i*n) * sz) + (n*n * 8)
                #the address should be the offset of the current row plus all the rows before it

        # print(y)

        # print(len(cache.blocks))
        # print(np.shape(cache.blocks))
        # print(cache.associativity)
        # print(cache.blocks[0][512])
    #     start = timeit.default_timer()
    #     for jj in tqdm(range(0, n, blocking_factor)):
    #         print(f"Actually iterating {jj}")
    #         for kk in range (0, n, blocking_factor):
    #             for i in range(0, n):
    #                 j = jj
    #                 j_stop = min(jj+blocking_factor, n)
    #                 k = kk 
    #                 k_stop = min(kk+blocking_factor, n)
    #                 for j in range(0, j_stop):
    #                     register0 = 0 
    #                     for k in range(0, k_stop):
    #                         register1 = myCpu.loadDouble(int(y[i][k]))
    #                         register2 = myCpu.loadDouble(int(z[k][j]))
    #                         register3 = myCpu.multDouble(register1, register2)
    #                         register0 += register3 
    #                     #print(f"ist iterating j{j}")
    #                     register4 = myCpu.loadDouble(int(x[i][j]))
    #                     register4 += register0
    #                     myCpu.storeDouble(int(x[i][j]), register4)
    #     stop = timeit.default_timer()
    #     print(f"Time Elapsed: {stop-start}")

    print(f"Instruction Count: {myCpu.instruction_count}")
    print(f"Cache.readhit: {cache.ReadHit}")
    print(f"Cache.readmiss: {cache.ReadMiss}")
    print(f"Cache.writehit: {cache.WriteHit}")
    print(f"Cache.writemiss: {cache.WriteMiss}") 

    print("INPUTS =======================================")
    print(f"Ram Size =                      {ram_num_blocks*input_block_size} bytes")
    print(f"Cache Size =                    {input_c} bytes")
    print(f"Block Size =                    {input_block_size} bytes")
    print(f"Total Blocks in Cache =         {Cache.numBlocks}")
    print(f"Associativity =                 {Cache.associativity}")
    print(f"Number of Sets =                {Cache.numSets}")
    print(f"Replacement Policy =            {input_r}")
    print(f"Algorithm =                     {input_a}")
    print(f"MXM Blocking Factor =           {input_f}")
    print(f"Matrix or Vector dimension =    {input_d}")



    # print(cache.associativity)
    # print(cache.blocks)
    # # print(f"Read Hits: {cache.ReadHit} / Read Miss: {cache.ReadMiss}")
    # # print(f"Write Hit: {cache.WriteHit} / Write Miss: {cache.WriteMiss}")
    # test_block = DataBlock()
    # print(isinstance(test_block, DataBlock))
    # test_block.b_data = np.arange(0, 64, 8, dtype=float)
    # my_test_block = (Address(64), test_block)
    # #my_block = CacheBlock(0, Address(64), test_block) 
    # #print("MyBlock")
    # #print(my_block.datablock)
    # cache.ram.setBlock(Address(72), test_block)
    # print("Ram Data: after Set Block")
    # print(cache.ram.data)
    # print("Cache Get Block Result(should match)")
    # print(cache.blocks)
    # #print(cache.getBlock(Address(100)))
    # print(cache.getDouble(100)) # Address(100) will be a different address object from Address(100)
    # print("Cache Blocks")
    # print(cache.blocks)
    # print(cache.getDouble(84))
    # print("Cache Blocks")
    # print(cache.blocks)
    # cache.setDouble(100, 15)
    # print("Cache Blocks")
    # print(cache.blocks)
    # cache.getDouble(200)
    # print("Cache Blocks")
    # print(cache.blocks)
    # cache.setDouble(0, 15) 
    # cache.setDouble(16, 32)
    # print("Cache Blocks")
    # print(cache.blocks)
    # print(f"Read Hits: {cache.ReadHit} / Read Miss: {cache.ReadMiss}")
    # print(f"Write Hit: {cache.WriteHit} / Write Miss: {cache.WriteMiss}")
    #corrct result should be 1/2, 2/1 

    #direct mapped cache, cache getblock from memory tests 
    # print("RAM data")
    # print(ram.data)
    # cache = Cache(ram)
    # test_block = DataBlock()
    # test_block.b_data = np.arange(0, 64, 8, dtype=float)
    # my_block = (Address(64), test_block) 
    # print("MyBlock")
    # print(my_block)
    # cache.ram.setBlock(Address(72), test_block)
    # print("Ram Data: after Set Block")
    # print(cache.ram.data)
    # print("Cache Get BLock Result(should match)")
    # print(cache.getBlock(Address(100)))
    #change the class attribute of block size to whatever the input is 

    

    
    

    #for set associavtivity 2 
    # print(cache.blocks[1][1].is_valid())
    # print(cache.blocks)
    # print("Original block data (should be null)")
    # cache.blocks[1][1].print_block_data()
    # print("my block\n")
    # print(my_block)
    # cache.setBlock(my_block[0], test_block)
    # print("data of the new set block(should match info from my_block") # for some when we set blocks the address is not updating 
    # cache.blocks[1][1].print_block_data() #because of random replacement assignment, it is either in this block or the other one 
    # # print(cache.blocks)
    # print("Get Block return value")
    # print(cache.getBlock(Address(64)))

    # cache.setBlock(Address(84), my_block[1])
    # my_double = cache.getDouble(100)
    # cache.setDouble(100, 420)
    # print(my_double)
    # # print("old cache blocks")
    # # print(cache.blocks)
    # print("my block")
    # print(my_block)
    # print("my_block data")
    # print(my_block[1].b_data)
    # #cache.blocks[0][1] = my_block
    # cache.setBlock(84, my_block[1])
    # print("new cache blocks")
    # print(cache.blocks)
    # gotten_block = cache.getBlock(72)
    #indexing error happens because you pass in the block not as a tuple without the address
    #need to figure out some way to fix 

   # print("get_block")
    #print(gotten_block)
    # print(test_block)
    # # print(test_block.b_data)
    # # test_block.b_data[3] = 12345
    # # print(test_block.b_data)

    """Address Level Testing"""
    #my_address = Address(1255)
    # print("Direct Mapped")
    # DataBlock.size = 64
    # Cache.numSets = 1 
    # print("index:")
    # print(my_address.get_index())
    # print("offset:")
    # print(my_address.get_offset())
    # print("tag:")
    # print(my_address.get_tag())

    # print("2-way set associate")
    # DataBlock.size = 64
    # Cache.associativity = 2 
    # Cache.numSets = Cache.numBlocks // Cache.associativity ## this is done manually right now but need to figure out how to update 
    # print("index:")
    # print(my_address.get_index())
    # print("offset:")
    # print(my_address.get_offset())
    # print("tag:")
    # print(my_address.get_tag())

    # print("8-way set associate")
    # Cache.associativity = 8 
    # Cache.numSets = Cache.numBlocks // Cache.associativity
    # print("index:")
    # print(my_address.get_index())
    # print("offset:")
    # print(my_address.get_offset())
    # print("tag:")
    # print(my_address.get_tag())
    # """ Block Level Testing""" 


    # # test_block_2 = DataBlock()
    # # test_block_2.b_data = np.arange(0, 8)
    # #print(DataBlock.size)
    # #print(test_block.size)
    # #print(test_block.b_data[0].address)

    # """Cache Testing"""
    # cache = Cache()
    # print(cache.blocks)
    # cache.blocks[0] = test_block
    # print(cache.blocks)

    # """Ram Level Testing"""
    # RAM = Ram(8)
    # # print(RAM.data)
    # # block_1 = RAM.getBlock(16)
    # # print(block_1.b_data)
    
    # # RAM.setBlock(100, test_block_2)
    # # print(RAM.getBlock(100).b_data)
    # #
    # print(RAM.data[1].get_address(8))
    # #print(RAM.data[0].b_data[1].address)