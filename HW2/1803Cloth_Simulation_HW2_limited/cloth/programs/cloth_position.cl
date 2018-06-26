__kernel
void cloth_position(
    __global float4* pos_in, __global float4* pos_out,
    __global float4* vel_in, __global float4* vel_out,
    __local float4* local_data,
    float3 Gravity,
    float ParticleMass,
    float ParticleInvMass,
    float SpringK,
    float RestLengthHoriz,
    float RestLengthVert,
    float RestLengthDiag,
    float DeltaT,
    float DampingConst) {
	//int idx = get_global_id(0) + get_global_size(0) * get_global_id(1);

	
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);
	int global_size_x = get_global_size(0);
	int global_size_y = get_global_size(1);
	int global_size = global_size_x * global_size_y;
	int global_index = global_y * global_size_x + global_x;
	
	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	int local_size_x = get_local_size(0);
	int local_size_y = get_local_size(1);
	int local_size = local_size_x * local_size_y;
	int local_index = local_y * local_size_x + local_x;
		
	int local_data_x = local_x + 1;
	int local_data_y = local_y + 1;
	int local_data_size_x = local_size_x + 2;
	int local_data_size_y = local_size_y + 2;
	int local_data_size = local_data_size_x * local_data_size_y;	
	int local_data_index = local_data_y * local_data_size_x + local_data_x;
	
	
	bool left_border_flag = (local_x == 0 && global_x > 0);
	bool right_border_flag = (local_x == local_size_x - 1 && global_x < global_size_x - 1);

	bool bottom_border_flag = (local_y == 0 && global_y > 0);
	bool top_border_flag = (local_y == local_size_y - 1 && global_y < global_size_y - 1);


	local_data[local_data_index] = pos_in[global_index];

	if (left_border_flag) {
		local_data[local_data_y * local_data_size_x + local_data_x - 1] = pos_in[global_y * global_size_x + global_x - 1];

		if (bottom_border_flag)
			local_data[(local_data_y - 1) * local_data_size_x + local_data_x - 1] = pos_in[(global_y - 1) * global_size_x + global_x - 1];
		if (top_border_flag)
			local_data[(local_data_y + 1) * local_data_size_x + local_data_x - 1] = pos_in[(global_y + 1) * global_size_x + global_x - 1];
	}
	else if (right_border_flag) {
		local_data[local_data_y * local_data_size_x + local_data_x + 1] = pos_in[global_y * global_size_x + global_x + 1];

		if (bottom_border_flag)
			local_data[(local_data_y - 1) * local_data_size_x + local_data_x + 1] = pos_in[(global_y - 1) * global_size_x + global_x + 1];
		if (top_border_flag)
			local_data[(local_data_y + 1) * local_data_size_x + local_data_x + 1] = pos_in[(global_y + 1) * global_size_x + global_x + 1];
	}

	if (bottom_border_flag) {
		local_data[(local_data_y - 1) * local_data_size_x + local_data_x] = pos_in[(global_y - 1) * global_size_x + global_x];
	}
	else if (top_border_flag) {
		local_data[(local_data_y + 1) * local_data_size_x + local_data_x] = pos_in[(global_y + 1) * global_size_x + global_x];
	}


	barrier(CLK_LOCAL_MEM_FENCE);

	float4 Fs;
	float4 r;

	r = local_data[local_data_y * local_data_size_x + local_data_x + 1] - local_data[local_data_y * local_data_size_x + local_data_x]; //	i+1, j & i, j
	Fs = (fabs(r) - RestLengthHoriz) * normalize(r);
	r = local_data[local_data_y * local_data_size_x + local_data_x - 1] - local_data[local_data_y * local_data_size_x + local_data_x]; //	i-1, j & i, j
	Fs += (fabs(r) - RestLengthHoriz) * normalize(r);
	r = local_data[(local_data_y + 1) * local_data_size_x + local_data_x] - local_data[local_data_y * local_data_size_x + local_data_x]; //	i, j+1 & i, j
	Fs += (fabs(r) - RestLengthVert) * normalize(r);
	r = local_data[(local_data_y - 1) * local_data_size_x + local_data_x] - local_data[local_data_y * local_data_size_x + local_data_x]; //	i, j-1 & i, j
	Fs += (fabs(r) - RestLengthVert) * normalize(r);

	r = local_data[(local_data_y - 1) * local_data_size_x + local_data_x - 1] - local_data[local_data_y * local_data_size_x + local_data_x]; //	i-1, j-1 & i, j
	Fs += (fabs(r) - RestLengthDiag) * normalize(r);
	r = local_data[(local_data_y + 1) * local_data_size_x + local_data_x - 1] - local_data[local_data_y * local_data_size_x + local_data_x]; //	i-1, j+1 & i, j
	Fs += (fabs(r) - RestLengthDiag) * normalize(r);
	r = local_data[(local_data_y - 1) * local_data_size_x + local_data_x + 1] - local_data[local_data_y * local_data_size_x + local_data_x]; //	i+1, j-1 & i, j
	Fs += (fabs(r) - RestLengthDiag) * normalize(r);
	r = local_data[(local_data_y + 1) * local_data_size_x + local_data_x + 1] - local_data[local_data_y * local_data_size_x + local_data_x]; //	i+1, j+1 & i, j
	Fs += (fabs(r) - RestLengthDiag) * normalize(r);
	Fs *= SpringK;
	
	float4 Fg = ParticleMass * (float4)(Gravity.x, Gravity.y, Gravity.z, 0);
	float4 Fd = -DampingConst * vel_in[global_index];

	float4 F = Fs + Fg + Fd;
	
	vel_out[global_index] = vel_in[global_index] + F * ParticleInvMass * DeltaT;
	pos_out[global_index] = pos_in[global_index] + vel_in[global_index] * DeltaT;

	if (global_y == global_size_y - 1) {
		pos_out[global_index] = pos_in[global_index];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}
