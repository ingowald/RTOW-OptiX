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

#include "material.h"
#include "prd.h"
#include "sampling.h"

/*! the implicit state's ray we will intersect against */
rtDeclareVariable(optix::Ray, ray,   rtCurrentRay, );
/*! the per ray data we operate on */
rtDeclareVariable(PerRayData, prd,   rtPayload, );
rtDeclareVariable(rtObject,   world, , );



/*! the attributes we use to communicate between intersection programs and hit program */
rtDeclareVariable(float3, hit_rec_normal, attribute hit_rec_normal, );
rtDeclareVariable(float3, hit_rec_p, attribute hit_rec_p, );


/*! and finally - that particular material's parameters */
rtDeclareVariable(float3, albedo, , );
rtDeclareVariable(float,  fuzz,   , );


/*! the actual scatter function - in Pete's reference code, that's a
  virtual function, but since we have a different function per program
  we do not need this here */
inline __device__ bool scatter(const optix::Ray &ray_in,
                               DRand48 &rndState,
                               vec3f &scattered_origin,
                               vec3f &scattered_direction,
                               vec3f &attenuation)
{
  float3 hit_pt_world = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_rec_p);
  float3 normal_world = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, hit_rec_normal));

  float3 hrn = normal_world;
  vec3f reflected = reflect(unit_vector(ray_in.direction),normal_world);
  scattered_origin    = hit_pt_world;
  scattered_direction = (reflected+fuzz*random_in_unit_sphere(rndState));
  attenuation         = vec3f(1.f);//albedo;
  return (dot(scattered_direction, hrn) > 0.f);
}

RT_PROGRAM void closest_hit()
{
  prd.out.scatterEvent
    = scatter(ray,
              *prd.in.randState,
              prd.out.scattered_origin,
              prd.out.scattered_direction,
              prd.out.attenuation)
    ? rayGotBounced
    : rayGotCancelled;
}
