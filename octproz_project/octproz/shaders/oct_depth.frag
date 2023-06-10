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


#version 130

out vec4 a_colour;

uniform mat4 ViewMatrix;
uniform mat3 NormalMatrix;

uniform float focal_length;
uniform float aspect_ratio;
uniform vec2 viewport_size;
uniform vec3 ray_origin;
uniform vec3 top;
uniform vec3 bottom;

uniform vec3 background_colour;
uniform vec3 material_colour;
uniform vec3 light_position;

uniform float step_length;
uniform float threshold;

uniform sampler3D volume;
uniform sampler3D depthTexture;
uniform sampler2D jitter;
uniform sampler1D lut;

uniform float gamma;
uniform float alpha_exponent;
uniform bool shading_enabled;
uniform bool lut_enabled;

// Ray
struct Ray {
	vec3 origin;
	vec3 direction;
};

// Axis-aligned bounding box
struct AABB {
	vec3 top;
	vec3 bottom;
};

//from https://iquilezles.org/articles/normalsSDF/
vec3 normal(vec3 position)
{
	float h = 0.005;
	vec3 n = vec3(0.0, 0.0, 0.0);
	for(int i=0; i<4; i++) {
		vec3 e =0.577350269*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
		n += e*texture(volume, position+e*h).r;
	}
	return -normalize(n);
}

vec3 normal_smooth(vec3 position, const int smoothing_factor)
{
	float delta = 0.001;
	int counter = 0;
	vec3 averaged_normal;
	const int n = smoothing_factor;
	for(int x = -1*n; x <= n; x++) {
		for(int y = -1*n; y <= n; y++) {
			for(int z = -1*n; z <= n; z++) {
				vec3 deltaPos = position + vec3(x*delta, y*delta, z*delta);
				averaged_normal += normal(deltaPos);
				counter++;
			}
		}
	}
	return normalize(averaged_normal /= counter);
}

// Slab method for ray-box intersection
void ray_box_intersection(Ray ray, AABB box, out float t_0, out float t_1)
{
	vec3 direction_inv = 1.0 / ray.direction;
	vec3 t_top = direction_inv * (box.top - ray.origin);
	vec3 t_bottom = direction_inv * (box.bottom - ray.origin);
	vec3 t_min = min(t_top, t_bottom);
	vec2 t = max(t_min.xx, t_min.yz);
	t_0 = max(0.0, max(t.x, t.y));
	vec3 t_max = max(t_top, t_bottom);
	t = min(t_max.xx, t_max.yz);
	t_1 = min(t.x, t.y);
}

// A very simple colour transfer function
vec4 colour_transfer(float intensity, float exponent)
{
	vec3 high = vec3(1.0, 1.0, 1.0);
	vec3 low = vec3(0.0, 0.0, 0.0);
	float alpha = pow(intensity, exponent);
	return vec4(intensity * high + (1.0 - intensity) * low, alpha);
}

// Blinn-Phong shading
vec3 shading(vec3 colour, vec3 position, vec3 ray)
{
	vec3 L = normalize(light_position - position);
	vec3 V = -normalize(ray);
	vec3 N = normal(position);
	vec3 H = normalize(L + V);

	float Ia = 0.75;
	float Id = 0.5 * max(0, dot(N, L));
	float Is = 1.0 * pow(max(0, dot(N, H)), 600);

	return (Ia + Id) * colour.rgb + Is * vec3(1.0);
}

void main()
{
	vec3 ray_direction;
	ray_direction.xy = 2.0 * gl_FragCoord.xy / viewport_size - 1.0;
	ray_direction.x *= aspect_ratio;
	ray_direction.z = -focal_length;
	ray_direction = (vec4(ray_direction, 0) * ViewMatrix).xyz;

	float t_0, t_1;
	Ray casting_ray = Ray(ray_origin, ray_direction);
	AABB bounding_box = AABB(top, bottom);
	ray_box_intersection(casting_ray, bounding_box, t_0, t_1);

	vec3 ray_start = (ray_origin + ray_direction * t_0 - bottom) / (top - bottom);
	vec3 ray_stop = (ray_origin + ray_direction * t_1 - bottom) / (top - bottom);

	vec3 ray = ray_stop - ray_start;
	float ray_length = length(ray);
	vec3 step_vector = step_length * ray / ray_length;

	// Random jitter
	ray_start += step_vector * texture(jitter, gl_FragCoord.xy / viewport_size).r;
	ray_stop += step_vector * texture(jitter, gl_FragCoord.xy / viewport_size).r;

	vec3 position = ray_stop;
	vec4 colour = vec4(0.0);
	float depthPositionOld = 1.0;

	//iterate over volume
	while (ray_length > 0) {

		 //get intensity and OCT depth at current position inside volume
		float intensity = texture(volume, position).r;
		float depthPosition = texture(depthTexture, position).r;
		float depthPositionDelta = abs(depthPosition-depthPositionOld); // This is used to avoid color artifacts due to discontinuities close to the surface. The surface itself is assigned a depth value of 1.0, while the area above it is assigned a value of 0.0.
		depthPositionOld = depthPosition;

		if(intensity > threshold && intensity < 0.9 && depthPosition > 0.1 && depthPositionDelta < 1.01*step_length){
			//colour transfer function
			vec4 c;
			if(lut_enabled){
				//c = texture(lut, position.b); //oct depth coloring without sample surface detection
				c = texture(lut, depthPosition-0.05 ); //oct depth coloring with sample surface as refernce
				c.a = pow(intensity, alpha_exponent);
			} else {
				//c = colour_transfer((0.2*intensity+0.8*depthPosition)/2.0, alpha_exponent);
				c = colour_transfer((depthPosition), alpha_exponent);
			}

			//alpha-blending
			colour.rgb = c.a * c.rgb + (1 - c.a) * colour.a * colour.rgb;
			colour.a = c.a + (1 - c.a) * colour.a;
			colour.rgb = colour.rgb/colour.a;


			if(shading_enabled){
				colour.rgb = shading(colour.rgb, position, ray);
			}
		}
		ray_length -= step_length;
		position -= step_vector;
	}

	//blend background
	colour.rgb = colour.a * colour.rgb + (1 - colour.a) * pow(background_colour, vec3(gamma)).rgb;
	colour.a = 1.0;

	//gamma correction
	a_colour.rgb = pow(colour.rgb, vec3(1.0 / gamma));
	a_colour.a = colour.a;
}
