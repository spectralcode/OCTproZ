//This file is a modified version of code originally created by Martino Pilia, please see: https://github.com/m-pilia/volume-raycasting

/**
**  This file is part of OCTproZ.
**  OCTproZ is an open source software for processig of optical
**  coherence tomography (OCT) raw data.
**  Copyright (C) 2019-2022 Miroslav Zabic
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

/*
 * Copyright © 2018 Martino Pilia <martino.pilia@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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
uniform float depth_weight;

uniform sampler3D volume;
uniform sampler2D jitter;
uniform sampler1D lut;

uniform float gamma;
uniform float alpha_exponent;
uniform bool shading_enabled;
uniform bool lut_enabled;

struct Ray {
	vec3 origin;
	vec3 direction;
};

//axis-aligned bounding box
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

//slab method for ray-box intersection
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

vec4 colour_transfer(float intensity, float exponent)
{
	vec3 high = vec3(1.0, 1.0, 1.0);
	vec3 low = vec3(0.1, 0.0, 0.2);
	float alpha = pow(intensity, exponent);
	return vec4(intensity * high + (1.0 - intensity) * low, alpha);
}

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

	//random jitter
	ray_start += step_vector * texture(jitter, gl_FragCoord.xy / viewport_size).r;
	vec3 position = ray_start;
	vec3 position_at_max = position;

	float maximum_intensity = 0.0;
	float ray_length_at_max = 0;
	float initial_ray_lenght = ray_length;

	//ray march until reaching the end of the volume or until max possible intensity encountered (early ray termination)
	while (ray_length > 0 && maximum_intensity < 0.99) {

		float intensity = texture(volume, position).r;

		if (intensity > maximum_intensity && intensity > threshold) {
			maximum_intensity = intensity;
			ray_length_at_max = ray_length;
			position_at_max = position;
		}

		ray_length -= step_length;
		position += step_vector;
	}

	//transfer function
	vec4 colour;
	if(lut_enabled){
		colour= texture(lut, maximum_intensity);
		colour.a = pow(maximum_intensity, alpha_exponent);
	} else {;
		colour = colour_transfer(maximum_intensity, alpha_exponent);
	}

	//multiply MIP with depth component (inspired by Díaz Iriberri, José, and Pere Pau Vázquez Alcocer. "Depth-enhanced maximum intensity projection." 8th IEEE/EG International Symposium on Volume Graphics. 2010.)
	float depth = length(position_at_max-ray_start)/(length(ray_stop-ray_start));
	colour = colour*((1.0-depth_weight)+2.0*depth_weight*(1.0-depth));

	//blend background
	colour.rgb = colour.a * colour.rgb + (1 - colour.a) * pow(background_colour, vec3(gamma)).rgb;
	colour.a = 1.0;

	//gamma correction
	a_colour.rgb = pow(colour.rgb, vec3(1.0 / gamma));
	a_colour.a = colour.a;
}
