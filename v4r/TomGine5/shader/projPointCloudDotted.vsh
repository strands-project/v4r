/*
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (c) 2014, Simon Schreiberhuber
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author simon.schreiberhuber
 *
 */

#version 130

in vec2 pos;


out vec4 color;

uniform float depthScale;
uniform float pointSize;
uniform vec2 uv0;
uniform vec2 f;
uniform mat4 mvp;

uniform sampler2D depthTex;
uniform sampler2D rgbTex;


void main(){

    vec2 texRes= textureSize(depthTex,0);
    vec2 texPos=vec2(pos.x/texRes.x+(0.5/texRes.x),pos.y/texRes.y+(0.5/texRes.y));
    float depth=-texture(depthTex,texPos).x*depthScale;
    //depth=-1;
    color=texture(rgbTex,texPos);
    color.y=color.x;
    color.z=color.x;
    //color=vec4(0,1,0,1);
    vec3 pos3=vec3(-(pos.x-uv0.x)*depth/f.x,(pos.y-uv0.y)*depth/f.y,depth);
    //pos3=vec3(pos,0);
    gl_Position = mvp*vec4(pos3,1);

    gl_PointSize = pointSize;
    //gl_Position.y=-gl_Position.y;
}
