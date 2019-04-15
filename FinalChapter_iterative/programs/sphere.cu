// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_world.h>
#include "prd.h"

/*! the parameters that describe each individual sphere geometry */
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float,  radius, , );

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(float3, hit_rec_normal, attribute hit_rec_normal, );
rtDeclareVariable(float3, hit_rec_p, attribute hit_rec_p, );

/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd, rtPayload, );


// Program that performs the ray-sphere intersection
//
// note that this is here is a simple, but not necessarily most numerically
// stable ray-sphere intersection variant out there. There are more
// stable variants out there, but for now let's stick with the one that
// the reference code used.
RT_PROGRAM void hit_sphere(int pid)
{
  // See Ch. 7: Precision Improvements for Ray/Sphere Intersection
  // p.87 of Ray Tracing Gems, edited by Eric Haines and Tomas Akenine-Moller, Apress 2019.
  // http://www.realtimerendering.com/raytracinggems/
  const float3 d = ray.direction;
  const float3 f = ray.origin - center; //TODO: center assumed to be zero, we can simplify this
  const float  a = dot(d, d);
  const float  b_prime = dot(-f, d);
  const float  r = radius;  //TODO: radius assumed to be one, we can simplify this
  const float3  l = f + (b_prime/a)*d; //not sure this is actually equivalent to l in the text?
  const float  discriminant = r*r - dot(l,l);
  
  if (discriminant < 0.f) return;

  float c = dot(f,f) - r*r;
  float q = b_prime + copysignf(sqrt(a*discriminant),b_prime);

  float t0 = c/q; 
  if (t0 < ray.tmax && t0 > ray.tmin) {
    if (rtPotentialIntersection(t0)) {
      hit_rec_p = ray.origin + t0 * ray.direction;
      /*hit_rec_normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD,(hit_rec_p - center) / radius));*/
      hit_rec_normal = hit_rec_p; //assumes center 0, radius 1. We transform to world in a later stage.
      rtReportIntersection(0);
    }
  }

  float t1 = q/a;
  if (t1 < ray.tmax && t1 > ray.tmin) {
    if (rtPotentialIntersection(t1)) {
      hit_rec_p = ray.origin + t1 * ray.direction;
      /*hit_rec_normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD,(hit_rec_p - center) / radius));*/
      hit_rec_normal = hit_rec_p; //assumes center 0, radius 1. We transform to world in a later stage.
      rtReportIntersection(0);
    }
  }
}

/*! returns the bounding box of the pid'th primitive
  in this gometry. Since we only have one sphere in this 
  program (we handle multiple spheres by having a different
  geometry per sphere), the'pid' parameter is ignored */
RT_PROGRAM void get_bounds(int pid, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = center - radius;
  aabb->m_max = center + radius;
}
