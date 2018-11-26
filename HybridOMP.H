#ifndef __HYBRIDOMP_H__
#define __HYBRIDOMP_H__

#ifdef __cplusplus
extern "C" {
#endif

int hyb_num_gpu_available(void);

int hyb_tid_to_device(int threadid);
void hyb_set_devices(void);

#ifdef __cplusplus
} //extern "C"
#endif


#endif //__HYBRIDOMP_H__
