// Propagator layout: Stack of nlayer 2d arrays of nx by ny float4 values: xr, xi, yr, yi
kernel void propagate(global float2 *complex_data, global const float2* propagator_x, global const float2* propagator_y, global float* kx2, global float* ky2, float k2max, int slice) {
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);
    size_t index = iy * get_global_size(0) + ix;
    float pxr, pxi, pyr, pyi;
    float2 w, t;
    
    if( kx2[ix] < k2max ) {
        pxr = propagator_x[ix].s0;
        pxi = propagator_x[ix].s1;

        if( (kx2[ix] + ky2[iy]) < k2max ) {
            pyr = propagator_y[iy].s0;
            pyi = propagator_y[iy].s1;
            w = complex_data[index];
            t.s0 = w.s0*pyr - w.s1*pyi;
            t.s1 = w.s0*pyi + w.s1*pyr;
            complex_data[index].s0 = t.s0*pxr - t.s1*pxi;
            complex_data[index].s1 = t.s0*pxi + t.s1*pxr;
        } else {
            complex_data[index] = 0.0f;
        }
        
    } else {
        complex_data[index] = 0.0f;
    }
    
}

kernel void transmit (global float2 *complex_data, global float2 *transmission_function, int ixoff, int iyoff, int trans_nx, int trans_ny) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int index = iy * get_global_size(0) + ix;
    int ixt, iyt;
    
    ixt = ix + ixoff;
    if (ixt >= trans_nx) {
        ixt = ixt - trans_nx;
    } else if (ixt < 0) {
        ixt = ixt + trans_nx;
    }
    
    iyt = iy + iyoff;
    if (iyt >= trans_ny) {
        iyt = iyt - trans_ny;
    } else if (iyt < 0) {
        iyt = iyt + trans_ny;
    }
    
    size_t index_trans = iyt * trans_nx + ixt;
    float2 pr = complex_data[index];
    float2 tr = transmission_function[index_trans];
    pr.s0 = pr.s0*tr.s0 - pr.s1*tr.s1;
    pr.s1 = pr.s0*tr.s1 + pr.s1*tr.s0;
    complex_data[index] = pr;
}

// temp has to be a buffer of at least probe-x-dimension * n_detect * sizeof(float)
kernel void sum_intensity(global float2 *probe, global float *kxp2, global float *kyp2, global float *k2min, global float *k2max, int n_detect, int probe_ny, global float *temp) {
    int ix = get_global_id(0);
    int probe_nx = get_global_size(0);
    
    // Initialize output for all detectors
    for(int idetect=0; idetect<n_detect; idetect++) {
        temp[idetect * probe_nx + ix] = 0;
    }
    
    for (int iy=0; iy<probe_ny; iy++) {
        float prr = probe[iy*probe_nx + ix].s0;
        float pri = probe[iy*probe_nx + ix].s1;
        float absolute = prr*prr + pri*pri;
        float k2 = kxp2[ix] + kyp2[iy];
        
        for(int idetect=0; idetect<n_detect; idetect++) {

            if ((k2 >= k2min[idetect]) && (k2 <= k2max[idetect])) {
                temp[idetect * probe_nx + ix] += absolute;
            }
            
        }
        
    }
    
}

kernel void prepare_probe_espread(float x, float y, float k2maxa, float k2maxb, float chi1, float chi2, float chi3, float xoff, float yoff, global float2 *probe_data, float wavlen, float dfa2, float dfa2phi, float dfa3, float dfa3phi, float df, global float *p_spat_freqx, global float *p_spat_freqy, global float *p_spat_freqx2, global float *p_spat_freqy2, global float *spat_freqx, global float *spat_freqy) {
    
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);
    size_t index = iy * get_global_size(0) + ix;
    
    size_t nx = get_global_size(0);
    size_t ny = get_global_size(1);
    
    float2 pd = probe_data[index];    
    float scale = 1.0/sqrt((float)nx*(float)ny);
    
    float pi = 4.0f * atan( 1.0f );
    
    float k2 = p_spat_freqx2[ix] + p_spat_freqy2[iy];
    
    float ktheta2 = k2*(wavlen*wavlen);
    float w = 2.0f*pi* ( xoff*p_spat_freqx[ix] + yoff*p_spat_freqy[iy] );
    float phi = atan2( spat_freqy[iy], spat_freqx[ix] );
    
    float chi = ktheta2*(-df+chi2 + dfa2*sin(2.0f*(phi-dfa2phi)))/2.0f;
    chi *= 2*pi/wavlen;
    chi -= w;
    
    if (fabs(k2-k2maxa) <= chi1) {
		pd.s0 = 0.5 * scale * cos(chi);
		pd.s1 = -0.5 * scale * sin(chi);
	} else if( (k2 >= k2maxa) && (k2 <= k2maxb) ) {
	    pd.s0 = scale * cos( chi );  /* real */
        pd.s1 = -scale * sin( chi );  /* imag */
    } else {
        pd.s0 = 0.0f;
        pd.s1 = 0.0f;
    }
    
    probe_data[index] = pd;
}

kernel void prepare_probe(float x, float y, float k2maxa, float k2maxb, float chi1, float chi2, float chi3, float xoff, float yoff, global float2 *probe_data, float wavlen, float dfa2, float dfa2phi, float dfa3, float dfa3phi, float df, global float *p_spat_freqx, global float *p_spat_freqy, global float *p_spat_freqx2, global float *p_spat_freqy2, global float *spat_freqx, global float *spat_freqy) {
    
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);
    size_t index = iy * get_global_size(0) + ix;
    
    float pi = 4.0f * atan( 1.0f );
    
    float2 pd = probe_data[index];
    
    float k2 = p_spat_freqx2[ix] + p_spat_freqy2[iy];
    if( (k2 >= k2maxa) && (k2 <= k2maxb) ) {
        float w = 2.0f*pi* ( xoff*p_spat_freqx[ix] + yoff*p_spat_freqy[iy] );
        float phi = atan2( spat_freqy[iy], spat_freqx[ix] );
        float chi = chi1*k2* ( (chi2 + chi3*k2)*k2 - df
                              + dfa2*sin( 2.0f*(phi-dfa2phi) )
                              + 2.0f * dfa3 * wavlen * sqrt(k2)*
                              sin( 3.0f*(phi-dfa3phi) )/3.0f );
        chi= - chi + w;
        
        pd.s0 = (float) cos( chi );
        pd.s1 = (float) sin( chi );
    } else {
        pd.s0 = 0.0f;
        pd.s1 = 0.0f;
    }
    probe_data[index] = pd;
}

kernel void scale_probe(global float2 *probe_data, float probe_sum) {
    float scale = (float) ( 1.0f/sqrt(probe_sum) );
    
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);
    size_t index = iy * get_global_size(0) + ix;
    
    float2 pd2 = probe_data[index];
    pd2.s0 *= scale;
    pd2.s1 *= scale;
    probe_data[index] = pd2;
}

kernel void gauss_multiply(global float2 *probe_data, float gausscale) {

	size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);
    size_t index = iy * get_global_size(0) + ix;
    
    size_t nx = get_global_size(0);
    size_t ny = get_global_size(1);
    
    float2 pd = probe_data[index]; 

	float r = exp(-((ix-nx/2.0f)*(ix-nx/2.0f)+(iy-ny/2.0f)*(iy-ny/2.0f))/((float)nx*(float)ny*gausscale));
	pd.s0 *= r;
	pd.s1 *= r;
		
	probe_data[index] = pd;
}
