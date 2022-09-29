#include <stdio.h>
#include <stdlib.h>
#include <nuttx/mtd/mtd.h>

/* Cache MMU block size */
#define MMU_BLOCK_SIZE          0x00010000
#define MMU_FLASH_MASK          (~(MMU_BLOCK_SIZE - 1))

#define INVALID_ADDR            0xFFFFFFFF
#define VADDR_MAX               0x403FFFFF // esp32 request this
#define NUM_SLOT_BIG            2
#define SIZE_SLOT_BIG           0x50000 //320K
#define NUM_SLOT_MID            3
#define SIZE_SLOT_MID           0x30000 //192K
#define NUM_SLOT_SML            5
#define SIZE_SLOT_SML           0x20000 //128K


#if defined(CONFIG_INTERPRETERS_WAMR_RELO_XIP_DEBUG)
#define DEBUG
#endif

typedef struct {
    uint8_t  reserved;
    uint32_t paddr;
    int32_t slot;
} flash_page;

typedef struct {
    uint8_t  reserved;
    uint32_t vaddr;
    uint32_t slot_size;
    uint32_t text_size;
} vm_slot;

static flash_page g_flash_pages[] =
{
    /* free partition 1
     * 0x00B35000->0x0xBC0000(11,751,424->12,320,768)
     * 8 pages
     */
    { 0, 0x00B40000, -1}, // 0x00B35000 is not 64K aligned, so begin with 0x00B40000
    { 0, 0x00B50000, -1},
    { 0, 0x00B60000, -1},
    { 0, 0x00B70000, -1},
    { 0, 0x00B80000, -1},
    { 0, 0x00B90000, -1},
    { 0, 0x00BA0000, -1},
    { 0, 0x00BB0000, -1},

    /* free partition 2
     * 0x00BE0000->0x0x00C00000(12,451,840->12,582,912)
     * 2 pages
     */
    { 0, 0x00BE0000, -1},
    { 0, 0x00BF0000, -1},

    /* free partition 3
     * 0x00D00000->0x0x00E80000(13,631,488->15,204,352)
     * 24 pages
     */
    { 0, 0x00D00000, -1},
    { 0, 0x00D10000, -1},
    { 0, 0x00D20000, -1},
    { 0, 0x00D30000, -1},
    { 0, 0x00D40000, -1},
    { 0, 0x00D50000, -1},
    { 0, 0x00D60000, -1},
    { 0, 0x00D70000, -1},
    { 0, 0x00D80000, -1},
    { 0, 0x00D90000, -1},
    { 0, 0x00DA0000, -1},
    { 0, 0x00DB0000, -1},
    { 0, 0x00DC0000, -1},
    { 0, 0x00DD0000, -1},
    { 0, 0x00DE0000, -1},
    { 0, 0x00DF0000, -1},
    { 0, 0x00E00000, -1},
    { 0, 0x00E10000, -1},
    { 0, 0x00E20000, -1},
    { 0, 0x00E30000, -1},
    { 0, 0x00E40000, -1},
    { 0, 0x00E50000, -1},
    { 0, 0x00E60000, -1},
    { 0, 0x00E70000, -1}

    /* totally 34 pages */
};

#if 0
/*  10 slots with 128K averate size */
static vm_slot g_vm_slots[] =
{
    /* Group 1, 128K for each slot
     * 4 slots
     */
     {0, 0x40240000, 0x20000, 0},
     {0, 0x40260000, 0x20000, 0},
     {0, 0x40280000, 0x20000, 0},
     {0, 0x402A0000, 0x20000, 0},

    /* Group 2, 192K for each slot
     * 3 slots
     */
    {0, 0x402C0000, 0x30000, 0},
    {0, 0x402F0000, 0x30000, 0},
    {0, 0x40320000, 0x30000, 0},

    /* Group 3, 320K for each slot
     * 2 slots
     */
    {0, 0x40350000, 0x50000, 0},
    {0, 0x403A0000, 0x50000, 0}

    /* totally 11 slots, max support 11 wasm apps */
};
#endif

static vm_slot *g_vm_slots;
static uint32_t g_num_vm_slots = 0;
static uint32_t g_num_vm_slots_big = 0;
static uint32_t g_num_vm_slots_mid = 0;
static uint32_t g_num_vm_slots_sml = 0;
static int32_t esp32_init_vm_slots()
{
    int32_t res = 0;
    uint32_t addr_max = VADDR_MAX + 1;
    uint32_t addr_min = (CONFIG_INTERPRETERS_WAMR_RELO_XIP_VADDR_START + 1) & MMU_FLASH_MASK;
    uint32_t addr = addr_max;
    uint32_t next_addr=0;

    static bool run_once = false;
    if(run_once)
    {
        return res;
    }

    for(int i = 0; i<NUM_SLOT_BIG;i++)
    {
        next_addr = (addr - SIZE_SLOT_BIG) & MMU_FLASH_MASK;
        if(next_addr < addr_min)
        {
            goto reach_min_vaddr;
        }
        addr = next_addr;
        g_num_vm_slots_big++;
    }

    for(int i = 0; i<NUM_SLOT_MID;i++)
    {
        next_addr = (addr - SIZE_SLOT_MID) & MMU_FLASH_MASK;
        if(next_addr < addr_min)
        {
            goto reach_min_vaddr;
        }
        addr = next_addr;
        g_num_vm_slots_mid++;
    }

    for(int i = 0; i<NUM_SLOT_SML;i++)
    {
        next_addr = (addr - SIZE_SLOT_SML) & MMU_FLASH_MASK;
        if(next_addr < addr_min)
        {
            goto reach_min_vaddr;
        }
        addr = next_addr;
        g_num_vm_slots_sml++;
    }

reach_min_vaddr:
    g_num_vm_slots = g_num_vm_slots_big + g_num_vm_slots_mid + g_num_vm_slots_sml;
    g_vm_slots = malloc(sizeof(vm_slot) * g_num_vm_slots);
    if(g_vm_slots == NULL)
    {
        printf("[ERRPR], malloc failed for g_vm_slots\n");
        return -1;
    }
    /* init the slots with 0 */
    memset(g_vm_slots, 0, sizeof(vm_slot) * g_num_vm_slots);

#ifdef DEBUG
    printf("/*static vm_slot g_vm_slots[] = \n");
    printf("{\n");
    printf("Group 1, 128K for each slot\n");
    printf("%d slots:*/\n", g_num_vm_slots_sml);
#endif
    for(int i = 0; i<g_num_vm_slots_sml;i++)
    {
        g_vm_slots[i].reserved = 0;
        g_vm_slots[i].vaddr = addr;
        g_vm_slots[i].slot_size = SIZE_SLOT_SML;
        g_vm_slots[i].text_size = 0;
        addr += SIZE_SLOT_SML;
#ifdef DEBUG
        printf("    {%d, 0x%x, 0x%x, %d},\n",
            g_vm_slots[i].reserved,
            g_vm_slots[i].vaddr,
            g_vm_slots[i].slot_size,
            g_vm_slots[i].text_size);
#endif
    }

#ifdef DEBUG
    printf("/*Group 2, 196K for each slot\n");
    printf("%d slots:*/\n", g_num_vm_slots_mid);
#endif
    uint32_t re_base = g_num_vm_slots_sml;
    for(int i = 0; i<g_num_vm_slots_mid;i++)
    {
        g_vm_slots[i+re_base].reserved = 0;
        g_vm_slots[i+re_base].vaddr = addr;
        g_vm_slots[i+re_base].slot_size = SIZE_SLOT_MID;
        g_vm_slots[i+re_base].text_size = 0;
        addr += SIZE_SLOT_MID;
#ifdef DEBUG
        printf("    {%d, 0x%x, 0x%x, %d},\n",
            g_vm_slots[i+re_base].reserved,
            g_vm_slots[i+re_base].vaddr,
            g_vm_slots[i+re_base].slot_size,
            g_vm_slots[i+re_base].text_size);
#endif
    }

#ifdef DEBUG
    printf("/*Group 3, 320K for each slot\n");
    printf("%d slots:*/\n", g_num_vm_slots_big);
#endif
    re_base = g_num_vm_slots_sml + g_num_vm_slots_mid;
    for(int i = 0; i<g_num_vm_slots_big;i++)
    {
        g_vm_slots[i+re_base].reserved = 0;
        g_vm_slots[i+re_base].vaddr = addr;
        g_vm_slots[i+re_base].slot_size = SIZE_SLOT_BIG;
        g_vm_slots[i+re_base].text_size = 0;
        addr += SIZE_SLOT_BIG;
#ifdef DEBUG
        printf("    {%d, 0x%x, 0x%x, %d},\n",
            g_vm_slots[i+re_base].reserved,
            g_vm_slots[i+re_base].vaddr,
            g_vm_slots[i+re_base].slot_size,
            g_vm_slots[i+re_base].text_size);
#endif
    }

#ifdef DEBUG
    printf("   /*totally %d slots, max support %d wasm apps*/ \n", g_num_vm_slots, g_num_vm_slots);
    printf("}\n");
#endif
    run_once = true;

    return res;
}

/**
 * get next flash address available that is 64K page aligned
 *
 */
static int32_t esp32_request_flash_page(uint32_t slot)
{
    int pages_num = sizeof(g_flash_pages) / sizeof(flash_page);
    for (int i = 0; i<pages_num; i++)
    {
        if(!g_flash_pages[i].reserved) {
            g_flash_pages[i].reserved = 1;
            g_flash_pages[i].slot = slot;
            return i;
        }
    }

    return -1;
}

static int32_t esp32_release_flash_pages(uint32_t slot)
{
    int pages_num = sizeof(g_flash_pages) / sizeof(flash_page);
    for (int i = 0; i<pages_num; i++)
    {
        if(g_flash_pages[i].reserved && g_flash_pages[i].slot == slot) {
            g_flash_pages[i].reserved = 0;
            g_flash_pages[i].slot = -1;
        }
    }

    return 0;
}

static int32_t esp32_request_vram_slot(uint32_t size)
{
    int slots_num = g_num_vm_slots;
    for (int i = 0; i<slots_num; i++)
    {
#ifdef DEBUG
        printf("g_vm_slots[%d].reserved(%d), g_vm_slots[%d].slot_size(%d)\n", i, g_vm_slots[i].reserved, i, g_vm_slots[i].slot_size);
#endif
        if(!g_vm_slots[i].reserved && size <= g_vm_slots[i].slot_size) {
            g_vm_slots[i].reserved = 1;
            g_vm_slots[i].text_size = size;
            return i;
        }
    }

    printf("[ERROR], no vm space is available\n");
    return -1;
}

static uint32_t esp32_release_vram_slot(uint32_t slot_id)
{
    int slots_num = g_num_vm_slots;
    if (slot_id >= slots_num)
    {
        printf("[ERROR], not find vm space to release accoring to slot_id:%d\n", slot_id);
        return -1;
    }

    g_vm_slots[slot_id].reserved = 0;
    return 0;
}

static int32_t esp32_get_vram_slot_by_vaddr(uint32_t vaddr)
{
    int slots_num = g_num_vm_slots;
    for (int i = 0; i < slots_num; i++)
    {
        if(g_vm_slots[i].reserved && vaddr == g_vm_slots[i].vaddr) {
            return i;
        }
    }

    return -1;
}

/**
 * get next 64K page aligned virtual address adjacent to current.
 * for the virtual memory must be continous.
 *
 * return value < 0, error
 * return value = 0, reach the end of its request, stop .
 * return value > 0, get the next adjacent vaddr
 */
static uint32_t esp32_next_vaddr_in_slot(uint32_t slot_id, uint32_t vaddr)
{
    int slots_num = g_num_vm_slots;

    if (slot_id >= slots_num)
    {
        printf("[ERROR], invalid slot_id:%d\n", slot_id);
        return INVALID_ADDR;
    }

    uint32_t slot_end = g_vm_slots[slot_id].vaddr + g_vm_slots[slot_id].slot_size;
    if(vaddr < g_vm_slots[slot_id].vaddr || vaddr >= slot_end)
    {
        printf("[ERROR], vaddr(0x%x) is beyond the slot", vaddr);
        return INVALID_ADDR;
    }

    uint32_t next_vaddr = vaddr + MMU_BLOCK_SIZE;
    if(next_vaddr > slot_end)
    {
        printf("[ERROR], the next vaddr 0x%x is beyond the slot(%d) end:0x%x\n", next_vaddr, slot_id, slot_end);
        return INVALID_ADDR;
    }

    if(next_vaddr >= g_vm_slots[slot_id].vaddr + g_vm_slots[slot_id].text_size)
    {
        return 0;
    } else
    {
        return next_vaddr;
    }
}

static int32_t esp32_write_flash(const uint8_t *buffer, uint32_t len, uint32_t flash_addr)
{
    struct mtd_dev_s *mtd;
    int ret = 0;
    uint32_t flash_addr_in = flash_addr;

    mtd = esp32_spiflash_get_mtd();
    if(!mtd)
    {
        printf("[ERROR], Failed to get SPI flash MTD\n");
        ret = -1;
        goto error;
    }

    /* check the alignment */

    struct mtd_geometry_s geo;
    ret = MTD_IOCTL(mtd, MTDIOC_GEOMETRY, (unsigned long)(uintptr_t)&geo);
    if(ret < 0)
    {
        printf("[ERROR], Failed to get flash geo\n");
        ret = -2;
        goto error;
    }

    int block_start = (flash_addr + geo.erasesize - 1) / geo.erasesize;
    int block_num = (len + geo.erasesize - 1) / geo.erasesize;
    flash_addr = block_start * geo.erasesize;
    flash_addr = flash_addr & MMU_FLASH_MASK;

    if(flash_addr_in != flash_addr)
    {
        printf("[WARNING], the flash address request is not aligned with system\n");
    }

    ret = MTD_ERASE(mtd, block_start, block_num);
    if (ret < 0)
    {
        printf("[ERROR], MTD_ERASE failed, ret=%d\n", ret);
        ret = -3;
        goto error;
    }

    ret = MTD_WRITE(mtd, flash_addr, len, buffer);
    if (ret != len)
    {
        printf("[ERROR], Failed to write data to MTD\n");
        ret = -4;
        goto error;
    }

    return flash_addr;

error:
    return ret;
}

int32_t esp32_app_request_vram(uint32_t size)
{
#ifdef DEBUG
    printf("esp32_app_request_vram enter, size = %d\n", size);
#endif
    int32_t init = esp32_init_vm_slots();
    if(init < 0)
    {
        printf("[ERROR], esp32_init_vm_slots failed\n");
        return -1;
    }

    int32_t slot = esp32_request_vram_slot(size);
    if (slot < 0)
    {
        printf("[ERROR], os_request_vram failed\n");
        return -2;
    }
#ifdef DEBUG
    printf("the vaddr is: 0x%x, int slot[%d]\n", g_vm_slots[slot].vaddr, slot);
#endif
    return g_vm_slots[slot].vaddr;
}

int32_t esp32_app_release_vram(uint32_t vaddr)
{
    int32_t slot = esp32_get_vram_slot_by_vaddr(vaddr);
    if(slot < 0)
    {
        printf("[ERROR], vaddr(0x%x) is invalid or not allocated\n", vaddr);
        return -1;
    }

    esp32_release_vram_slot(slot);
    esp32_release_flash_pages(slot);

    return 0;
}

int32_t esp32_app_vmmap(void *vdest, const void *src, uint size)
{
    extern unsigned int cache_flash_mmu_set(int cpu_no, int pid, unsigned int vaddr, unsigned int paddr, int psize, int num);

    int32_t slot = esp32_get_vram_slot_by_vaddr(vdest);
    if(slot < 0)
    {
        printf("[ERROR], vdest(0x%x) is invalid or not allocated\n", (unsigned int)vdest);
        return -1;
    }
    uint32_t next_vaddr = (uint32_t)vdest;
    const void *next_src = src;

    while (next_vaddr > 0) {
        int32_t page_index = esp32_request_flash_page(slot);
        if(page_index < 0) {
            printf("[ERROR], no flash page is available\n");
            return -2;
        }
        uint32_t next_paddr = g_flash_pages[page_index].paddr;
        int ret = cache_flash_mmu_set(0, 0, next_vaddr, next_paddr, 64, 1);
        if (ret != 0)
        {
            printf("ERROR: cache_flash_mmu_set failed!\n");
            return -3;
        }
    #ifdef DEBUG
        printf("map from next_vaddr(0x%x) to flash addr(0x%x)\n", next_vaddr, next_paddr);
    #endif

        /* write one page, or remaining if less than one page*/
        uint32_t write_size = MMU_BLOCK_SIZE;
        if((next_src + MMU_BLOCK_SIZE) > (src + size)) {
            write_size = src + size - next_src;
        }
        esp32_write_flash(next_src, write_size, next_paddr);
        next_vaddr = esp32_next_vaddr_in_slot(slot, next_vaddr);
        if(next_vaddr == INVALID_ADDR)
        {
            printf("ERROR: no space in slot!\n");
            return -4;
        }
        next_src += MMU_BLOCK_SIZE;
    }

    return 0;
}
