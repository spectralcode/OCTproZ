/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2023 Miroslav Zabic
**
**  OCTproZ is free software: you can redistribute it and/or modify
**  it under the terms of the GNU General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  This program is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
**  GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License
**  along with this program. If not, see http://www.gnu.org/licenses/.
**
****
** Author:	Miroslav Zabic
** Contact:	zabic
**			at
**			spectralcode.de
****
**/

#version 430

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (binding = 0, r8) uniform image3D volume;
layout (binding = 3, r32f) uniform image3D depthTexture;

uniform vec3 volumeSize;
uniform float depthIntensityThreshold;

void main()
{
	ivec3 globalID = ivec3(gl_GlobalInvocationID.xyz);

	if (globalID.x >= volumeSize.x || globalID.y >= volumeSize.y || globalID.z >= volumeSize.z) {
		return; // return if outside the volume
	}

	// Initialize depth to 0
	imageStore(depthTexture, globalID, vec4(0.0, 0.0, 0.0, 0.0));

	float val = 0.0;
	float depthValue = 1.0;
	float deltaDepth = 1.0/float(volumeSize.z);
	int startPos = int(volumeSize.z-volumeSize.z/32);
	for(int i = startPos; i > 0; i--) {
		if(val > depthIntensityThreshold) {
			imageStore(depthTexture, ivec3(globalID.x, globalID.y, i), vec4(depthValue, 0.0, 0.0, 0.0));
			 depthValue = depthValue - deltaDepth;
		} else {
			val = float(imageLoad(volume, ivec3(globalID.x, globalID.y, i)).r);
		}
	}
}
