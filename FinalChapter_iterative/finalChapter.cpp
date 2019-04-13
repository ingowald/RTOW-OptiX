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

// ooawe
#include "programs/vec.h"
#include "savePPM.h"
// optix
#include <optix.h>
#include <optixu/optixpp.h>
// std
#define _USE_MATH_DEFINES 1
#include <math.h>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <optixu/optixu_matrix_namespace.h>


optix::Context g_context;

/*! the precompiled programs/raygen.cu code (in ptx) that our
  cmake magic will precompile (to ptx) and link to the generated
  executable (ie, we can simply declare and usethis here as
  'extern'.  */
extern "C" const char embedded_sphere_programs[];
extern "C" const char embedded_raygen_program[];
extern "C" const char embedded_miss_program[];
extern "C" const char embedded_metal_programs[];
extern "C" const char embedded_dielectric_programs[];
extern "C" const char embedded_lambertian_programs[];

union raw_float {
	u_char buffer[4];
	float number;
};

float rnd()
{
  // static std::random_device rd;  //Will be used to obtain a seed for the random number engine
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

/*! abstraction for a material that can create, and parameterize,
  a newly created GI's material and closest hit program */
struct Material {
  virtual void assignTo(optix::GeometryInstance gi) const = 0;
};

/*! host side code for the "Lambertian" material; the actual
  sampling code is in the programs/lambertian.cu closest hit program */
struct Lambertian : public Material {
  /*! constructor */
  Lambertian(const vec3f &albedo) : albedo(albedo) {}
  /* create optix material, and assign mat and mat values to geom instance */
  virtual void assignTo(optix::GeometryInstance gi) const override {
    optix::Material mat = g_context->createMaterial();
    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString
                              (embedded_lambertian_programs,
                               "closest_hit"));
    gi->setMaterial(/*ray type:*/0, mat);
    gi["albedo"]->set3fv(&albedo.x);
  }
  const vec3f albedo;
};

/*! host side code for the "Metal" material; the actual
  sampling code is in the programs/metal.cu closest hit program */
struct Metal : public Material {
  /*! constructor */
  Metal(const vec3f &albedo, const float fuzz) : albedo(albedo), fuzz(fuzz) {}
  /* create optix material, and assign mat and mat values to geom instance */
  virtual void assignTo(optix::GeometryInstance gi) const override {
    optix::Material mat = g_context->createMaterial();
    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString
                              (embedded_metal_programs,
                               "closest_hit"));
    gi->setMaterial(/*ray type:*/0, mat);
    gi["albedo"]->set3fv(&albedo.x);
    gi["fuzz"]->setFloat(fuzz);
  }
  const vec3f albedo;
  const float fuzz;
};

/*! host side code for the "Dielectric" material; the actual
  sampling code is in the programs/dielectric.cu closest hit program */
struct Dielectric : public Material {
  /*! constructor */
  Dielectric(const float ref_idx) : ref_idx(ref_idx) {}
  /* create optix material, and assign mat and mat values to geom instance */
  virtual void assignTo(optix::GeometryInstance gi) const override {
    optix::Material mat = g_context->createMaterial();
    mat->setClosestHitProgram(0, g_context->createProgramFromPTXString
                              (embedded_dielectric_programs,
                               "closest_hit"));
    gi->setMaterial(/*ray type:*/0, mat);
    gi["ref_idx"]->setFloat(ref_idx);
  }
  const float ref_idx;
};



/*
 *optix::GeometryGroup createScene()
 *{ 
 *  // first, create all geometry instances (GIs), and, for now,
 *  // store them in a std::vector. For ease of reference, I'll
 *  // stick wit the 'd_list' and 'd_world' names used in the
 *  // reference C++ and CUDA codes.
 *  std::vector<optix::GeometryInstance> d_list;
 *
 *  d_list.push_back(createSphere(vec3f(0.f, -1000.0f, -1.f), 1000.f,
 *                                Lambertian(vec3f(0.5f, 0.5f, 0.5f))));
 *
 *  for (int a = -11; a < 11; a++) {
 *    for (int b = -11; b < 11; b++) {
 *      float choose_mat = rnd();
 *      vec3f center(a + rnd(), 0.2f, b + rnd());
 *      if (choose_mat < 0.8f) {
 *        d_list.push_back(createSphere(center, 0.2f,
 *                                      Lambertian(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd()))));
 *      }
 *      else if (choose_mat < 0.95f) {
 *        d_list.push_back(createSphere(center, 0.2f,
 *                                      //Metal(vec3f(0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd()), 0.5f*(1.0f + rnd())), 0.5f*rnd())));
 *                                      Lambertian(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd()))));
 *      }
 *      else {
 *        //d_list.push_back(createSphere(center, 0.2f, Dielectric(1.5f)));
 *        d_list.push_back(createSphere(center, 0.2f,
 *                                      Lambertian(vec3f(rnd()*rnd(), rnd()*rnd(), rnd()*rnd()))));
 *      }
 *    }
 *  }
 *  //d_list.push_back(createSphere(vec3f(0.f, 1.f, 0.f), 1.f, Dielectric(1.5f)));
 *  //d_list.push_back(createSphere(vec3f(-4.f, 1.f, 0.f), 1.f, Lambertian(vec3f(0.4f, 0.2f, 0.1f))));
 *  //d_list.push_back(createSphere(vec3f(4.f, 1.f, 0.f), 1.f, Metal(vec3f(0.7f, 0.6f, 0.5f), 0.0f)));
 *
 *  d_list.push_back(createSphere(vec3f(0.f, 1.f, 0.f), 1.f, Lambertian(vec3f(0.4f, 0.2f, 0.1f))));
 *  d_list.push_back(createSphere(vec3f(-4.f, 1.f, 0.f), 1.f, Lambertian(vec3f(0.4f, 0.2f, 0.1f))));
 *  d_list.push_back(createSphere(vec3f(4.f, 1.f, 0.f), 1.f, Lambertian(vec3f(0.7f, 0.6f, 0.5f))));
 *  
 *  // now, create the optix world that contains all these GIs
 *  optix::GeometryGroup d_world = g_context->createGeometryGroup();
 *  d_world->setAcceleration(g_context->createAcceleration("Bvh"));
 *  d_world->setChildCount((int)d_list.size());
 *  for (int i = 0; i < d_list.size(); i++)
 *    d_world->setChild(i, d_list[i]);
 *
 *  // that all we have to do, the rest is up to optix
 *  return d_world;
 *}
 */

optix::GeometryInstance createSphere(const vec3f &center, const float radius, const Material &material)
{
  optix::Geometry geometry = g_context->createGeometry();
  geometry->setPrimitiveCount(1);
  geometry->setBoundingBoxProgram
    (g_context->createProgramFromPTXString(embedded_sphere_programs, "get_bounds"));
  geometry->setIntersectionProgram
    (g_context->createProgramFromPTXString(embedded_sphere_programs, "hit_sphere"));
  geometry["center"]->setFloat(center.x,center.y,center.z);
  geometry["radius"]->setFloat(radius);

  optix::GeometryInstance gi = g_context->createGeometryInstance();
  gi->setGeometry(geometry);
  gi->setMaterialCount(1);
  material.assignTo(gi);
  return gi;
}


//create a unit sphere centered at the origin
optix::GeometryInstance createUnitSphere(const Material &material){
  return createSphere(vec3f(0.f, 0.f, 0.f), 1.0f, material);
}

//Assumes the transform is paired with a unit sphere.
optix::Transform createSphereXform(const vec3f &center, const float radius, const optix::GeometryGroup &gg)
{
  //create transform based on the given center and radius
  float sphereMatRaw[16] =
  {                    
    radius, 0.0f,   0.0f,   center.x,
    0.0f,   radius, 0.0f,   center.y, 
    0.0f,   0.0f,   radius, center.z,
    0.0f,   0.0f,   0.0f,   1.0f                                 
  };                                                              
  optix::Matrix4x4 matrixSphere(sphereMatRaw);   
  optix::Transform trSphere = g_context->createTransform();
  trSphere->setMatrix(false, matrixSphere.getData(),
      matrixSphere.inverse().getData());    
  
  trSphere->setChild(gg); 

  return trSphere;
}


// ----------------------File-reading functions -----------------------------------------
// All of these return a vector of vectors

// Returns a list of tensors. Each tensor has the 6 parts of a symmetric tensor and the corresponding x,y,z location of the tensor.
// Data is in the order Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, x, y, z
std::vector<std::vector<float> > raw_spiral_reader() {
		std::vector<std::vector<float> > tensors;
		// [0,1,2,3,4,5] = tensor data, [6,7,8] = x,y,z
		
	std::string line;
	std::ifstream csvfile("../dt-helix.raw");
	int count = 0;
	if(csvfile.is_open()) {
		int file_size = 7*38*39*40; // x=38,y=39,z=40, tensor=7 (first value indicates relevance)
		u_char *pos;
		raw_float rf;
		raw_float rfrev;
		//t=410412  // This is the total number of floats in the file. The thing is, it's less than the original number by approximately 4500
		//count=1641647 count/4=410411, file size=414960=7*38*39*40, total size in char=3319680
		u_char temp;
		// 1=confidence,2 Dxx, 3 Dxy, 4 Dxz, 5 Dyy, 6 Dyz, 7 Dzz
		//t changes the fastest, then x is fast, y is medium, and z is slow.
		int x,y,z,t;
		x=0;
		y=0;
		z=0;
		t=0;
		//axis mins:  NaN -2 -2 -2
		//axis maxs:  NaN 2 2 2
		int t_len = 7;
		int x_len = 38; // x/9.5 - 2 
		int y_len = 39; // y/9.75 - 2
		int z_len = 40; // z*0.1 - 2
		double zd,yd,xd;
		//while(count<200) {
		float tensor[7];
		while(!csvfile.eof()){//csvfile.good()){
			csvfile>>temp;
			rf.buffer[3-(count%4)] = temp;
			rfrev.buffer[count%4] = temp;
			//printf("%.2x\n",temp);
			if((count%4)==3){
				tensor[t]=rf.number;
				//tensor[t]=rfrev.number; // If it turns out we are doing things in the wrong direction, use rfrev.
				//printf("z,y,x,t=%d,%d,%d,%d\n",z,y,x,t);
				//printf("%f\n",rf.number);
				//std::cout<<std::endl;
				if(t==(t_len-1)){
					zd=double(z)*0.1;//-2;
					yd=double(y)/9.75;//-2;
					xd=double(x)/9.75;//-2;
					//printf("z,y,x,t=%f,%f,%f,%f\n",float(zd)-2,float(yd)-2,float(xd)-2);
					// Create a shape using the tensor data we saved. 
					// Only create it if t[0]>=0.5 && t[0]<=1.0
					//printf("t[0]=%f\n",tensor[0]);
					if(tensor[0]>=0.5&&tensor[0]<=1.0){
						std::vector<float> tensor_data;
						tensor_data.push_back(tensor[0]);
						tensor_data.push_back(tensor[1]);
						tensor_data.push_back(tensor[2]);
						tensor_data.push_back(tensor[3]);
						tensor_data.push_back(tensor[4]);
						tensor_data.push_back(tensor[5]);
						tensor_data.push_back(tensor[6]);
						// Now add the point data. Note: We may need to reverse the order of x and z possibly
						tensor_data.push_back(float(xd)-2);
						tensor_data.push_back(float(yd)-2);
						tensor_data.push_back(float(zd)-2);
						tensors.push_back(tensor_data);
						
					}
				}
				if(y==(y_len-1)&&x==(x_len-1)&&t==(t_len-1)){
					t=0;
					x=0;
					y=0;
					z++;
				}
				else if(x==(x_len-1)&&t==(t_len-1)) {
					x=0;
					t=0;
					y++;
				}
				else if(t==(t_len-1)) {
					t=0;
					x++;
				}
				else{
					t++;
				}
			}
			count++;
		}
		//printf("t=%d\n",t);
		//printf("count=%d count/4=%d, total size=%d, total size in char=%d\n",count-1, (count-1)/4,file_size,file_size*8);
		csvfile.close();
	}
	else {
		std::cout<<"Unable to open spiral helix file";
	}
	
	return tensors;
}



// Reads a tensor.csv file and returns the tensor data as a list
// Each vector in the list of vectors has the data in the format Dxx,Dxy,Dxz,Dyx,Dyy,Dyz,Dzx,Dzy,Dzz,x,y,z
std::vector<std::vector<float> > pipe_tensors() {
	std::vector<std::vector<float> > tensors;
	
  std::string line;
  std::ifstream csvfile("../tensor.csv");
  int count =0; // This is just to limit the amount of the file we read for testing
  if(csvfile.is_open()){
	  //while(csvfile.good()? 
	  // while(getline(csvfile,line)){} // This line will allow us to read through the entire structure.
	  while(getline(csvfile,line)){
	  //while(count<5) {
		  //getline(csvfile,line);
		  //std::cout<<line<<'\n';
		  if(count>0){
			  std::vector<float> row;
			  std::string substr;
			  std::stringstream ss;
			  ss<<line;
			  while(ss.good()){
				  getline(ss,substr,',');
				  double temp = ::atof(substr.c_str());
				  row.push_back((float)temp);
			  }
			  float x,y,z;
			  //x = row[9];
			  //y = row[10];
			  //z = row[11];
			  //vec3f center(row[9],row[10],row[11]);
			  //t_list.push_back(createSphereXform(center,0.2f,ggDiffuse));
			  tensors.push_back(row);
		  }
		  count++;
	  }
	  csvfile.close();
  }
  return tensors;
	
}

optix::Group createScene()
{ 
  //Pre-create one geometry instance per material. 
  optix::GeometryInstance giDiffuseSphere = createUnitSphere(Lambertian(vec3f(0.5f, 0.5f, 0.5f)));
  optix::GeometryInstance giMetalSphere = createUnitSphere(Metal(vec3f(0.7f, 0.6f, 0.5f), 0.0f));
  optix::GeometryInstance giGlassSphere = createUnitSphere(Dielectric(1.5f));

  //Make a geometry group for each of the geometry instances created above. 
  //Build an acceleration structure for each.
  optix::Acceleration accDiffuse = g_context->createAcceleration("Bvh");
  optix::GeometryGroup ggDiffuse = g_context->createGeometryGroup();
  ggDiffuse->setAcceleration(accDiffuse);
  ggDiffuse->setChildCount(1);
  ggDiffuse->setChild(0, giDiffuseSphere);

  optix::Acceleration accMetal = g_context->createAcceleration("Bvh");
  optix::GeometryGroup ggMetal = g_context->createGeometryGroup();
  ggMetal->setAcceleration(accMetal);
  ggMetal->setChildCount(1);
  ggMetal->setChild(0, giMetalSphere);

  optix::Acceleration accGlass = g_context->createAcceleration("Bvh");
  optix::GeometryGroup ggGlass = g_context->createGeometryGroup();
  ggGlass->setAcceleration(accGlass);
  ggGlass->setChildCount(1);
  ggGlass->setChild(0, giGlassSphere);

  //Push the transform returned by createSphereXform to t_list
  std::vector<optix::Transform> t_list;
  
  // This is the plane the original balls rested on
  //t_list.push_back(createSphereXform(vec3f(0.f, -1000.0f, -1.f), 1000.f, ggDiffuse)); 
  
  // Get spiral vector tensors. 
  // Data is in the order Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, x, y, z. Each axis ranges from -2 to 2.
	std::vector<std::vector<float> > tensors2 = raw_spiral_reader();
	std::cout<<"Tensor vector length="<<tensors2.size();

	// Get the pipe tensors
	// Data is in the order Dxx, Dxy, Dxz, Dyx, Dyy, Dyz, Dzx, Dzy, Dzz, x, y, z
	std::vector<std::vector<float> > tensors = pipe_tensors();
	std::cout<<"Tensor vector length for the pipe="<<tensors.size();
	
	// **** Work on this
	if(tensors.size()>0){
		for(std::vector<std::vector<float> >::iterator it = tensors.begin(); it != tensors.end(); it++) {
			std::vector<float> row = *it;
			//std::cout << ' ' << row.size()<<std::endl; // This particular one is 10 items long
			//printf("%f %f %f %f %f %f %f %f",row[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],temp[7]);
			// Create the shape and add it to the list
			vec3f center(row[9],row[10],row[11]);
			t_list.push_back(createSphereXform(center,0.2f,ggDiffuse));
		}
	}
  // --------------Uplift this code later to main()-------
  /*
  std::string line;
  std::ifstream csvfile("../tensor.csv");
  int count =0; // This is just to limit the amount of the file we read for testing
  if(csvfile.is_open()){
	  //while(csvfile.good()? 
	  // while(getline(csvfile,line)){} // This line will allow us to read through the entire structure.
	  while(getline(csvfile,line)){
	  //while(count<5) {
		  //getline(csvfile,line);
		  //std::cout<<line<<'\n';
		  if(count>0){
			  std::vector<float> row;
			  std::string substr;
			  std::stringstream ss;
			  ss<<line;
			  while(ss.good()){
				  getline(ss,substr,',');
				  double temp = ::atof(substr.c_str());
				  row.push_back((float)temp);
			  }
			  float x,y,z;
			  x = row[9];
			  y = row[10];
			  z = row[11];
			  vec3f center(row[9],row[10],row[11]);
			  t_list.push_back(createSphereXform(center,0.2f,ggDiffuse));
		  }
		  count++;
	  }
	  csvfile.close();
  }
  */
  //------------------------------------------------------
  
	/*
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) {
        t_list.push_back(createSphereXform(center, 0.2f, ggDiffuse));
      }
      else if (choose_mat < 0.95f) {
        t_list.push_back(createSphereXform(center, 0.2f, ggMetal));
        //t_list.push_back(createSphereXform(center, 0.2f, ggDiffuse));
      }
      else {
        t_list.push_back(createSphereXform(center, 0.2f, ggGlass));
        //t_list.push_back(createSphereXform(center, 0.2f, ggDiffuse));
      }
    }
  }

  t_list.push_back(createSphereXform(vec3f(0.f, 1.f, 0.f), 1.f, ggGlass));
  //t_list.push_back(createSphereXform(vec3f(0.f, 1.f, 0.f), 1.f, ggDiffuse));
  t_list.push_back(createSphereXform(vec3f(-4.f, 1.f, 0.f), 1.f, ggDiffuse));
  t_list.push_back(createSphereXform(vec3f(4.f, 1.f, 0.f), 1.f, ggMetal));
  //t_list.push_back(createSphereXform(vec3f(4.f, 1.f, 0.f), 1.f, ggDiffuse));
*/
  //At the end, instead of instantiating a GeometryGroup d_world, instantiate a group t_world.
  //Add children to t_world in the same way that we added children to d_world.
  optix::Group t_world = g_context->createGroup();
  t_world->setAcceleration(g_context->createAcceleration("Bvh"));
  t_world->setChildCount((int)t_list.size());
  for (int i = 0; i < t_list.size(); i++)
    t_world->setChild(i, t_list[i]);

  // that is all we have to do, the rest is up to optix
  return t_world;
}

struct Camera {
  Camera(const vec3f &lookfrom, const vec3f &lookat, const vec3f &vup, 
         float vfov, float aspect, float aperture, float focus_dist) 
  { // vfov is top to bottom in degrees
    lens_radius = aperture / 2.0f;
    float theta = vfov * ((float)M_PI) / 180.0f;
    float half_height = tan(theta / 2.0f);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);
    lower_left_corner = origin - half_width * focus_dist*u - half_height * focus_dist*v - focus_dist * w;
    horizontal = 2.0f*half_width*focus_dist*u;
    vertical = 2.0f*half_height*focus_dist*v;
  }
	
  void set()
  {
    g_context["camera_lower_left_corner"]->set3fv(&lower_left_corner.x);
    g_context["camera_horizontal"]->set3fv(&horizontal.x);
    g_context["camera_vertical"]->set3fv(&vertical.x);
    g_context["camera_origin"]->set3fv(&origin.x);
    g_context["camera_u"]->set3fv(&u.x);
    g_context["camera_v"]->set3fv(&v.x);
    g_context["camera_w"]->set3fv(&w.x);
    g_context["camera_lens_radius"]->setFloat(lens_radius);
  }
  vec3f origin;
  vec3f lower_left_corner;
  vec3f horizontal;
  vec3f vertical;
  vec3f u, v, w;
  float lens_radius;
};


void renderFrame(int Nx, int Ny)
{
  // ... and validate everything before launch.
  g_context->validate();

  // now that everything is set up: launch that ray generation program
  g_context->launch(/*program ID:*/0,
                    /*launch dimensions:*/Nx, Ny);
}

optix::Buffer createFrameBuffer(int Nx, int Ny)
{
  // ... create an image - as a 2D buffer of float3's ...
  optix::Buffer pixelBuffer
    = g_context->createBuffer(RT_BUFFER_OUTPUT);
  pixelBuffer->setFormat(RT_FORMAT_FLOAT3);
  pixelBuffer->setSize(Nx, Ny);
  return pixelBuffer;
}

void setRayGenProgram()
{
  optix::Program rayGenAndBackgroundProgram
    = g_context->createProgramFromPTXString(embedded_raygen_program,
                                            "renderPixel");
  g_context->setEntryPointCount(1);
  g_context->setRayGenerationProgram(/*program ID:*/0, rayGenAndBackgroundProgram);
}

void setMissProgram()
{
  optix::Program missProgram
    = g_context->createProgramFromPTXString(embedded_miss_program,
                                            "miss_program");
  g_context->setMissProgram(/*program ID:*/0, missProgram);
}

int main(int ac, char **av)
{
  // before doing anything else: create a optix context
  g_context = optix::Context::create();
  g_context->setRayTypeCount(1);
  g_context->setStackSize( 3000 );
  
  // define some image size ...
  const size_t Nx = 1200, Ny = 800;

  // create - and set - the camera
  // Get the camera position and tell it to look at (0,0,0)
  //const vec3f lookfrom(13, 2, 3);
  //const vec3f lookfrom(0, 0, -10);
  //const vec3f lookat(0, 0, 0);
  //const vec3f vup(0,1,0);
  //float vfov = 20.0;
  // Camera params : const vec3f &lookfrom, const vec3f &lookat, const vec3f &vup, 
         //float vfov, float aspect, float aperture, float focus_dist
  // Position for tensor.csv
  //const vec3f lookfrom(33,19.7,22.85);
  //const vec3f lookat(7.6,13.69,12.4);
  //const vec3f vup(-0.082,0.936,-0.342);
  const vec3f lookfrom(1.149, -0.324, 1.027);
  const vec3f lookat(-42.929, -32.672, -18.724);
  const vec3f vup(0.644, -0.722, -0.255);
  float vfov = 45;
  Camera camera(lookfrom,
                lookat,
                /* up */ vup,//vec3f(0, 1, 0),
                /* fovy, in degrees */ vfov, //20.0,
                /* aspect */ float(Nx) / float(Ny),
                /* aperture */ 0.1f,
                /* dist to focus: */ 10.f);
  camera.set();

  // set the ray generation and miss shader program
  setRayGenProgram();
  setMissProgram();

  // create a frame buffer
  optix::Buffer fb = createFrameBuffer(Nx, Ny);
  g_context["fb"]->set(fb);

  // create the world to render
  //optix::GeometryGroup world = createScene();
  optix::Group world = createScene();
  g_context["world"]->set(world);

  const int numSamples = 128;
  g_context["numSamples"]->setInt(numSamples);

#if 1
  {
    // Note: this little piece of code (in the #if 1/#endif bracket)
    // is _NOT_ required for correctness; it's just been added to
    // factor our build time vs render time: In optix, the data
    // structure gets built 'on demand', which basically means it gets
    // built on the first launch after the scene got set up or
    // changed. Thus, if we create a 0-sized launch optix won't do any
    // rendering (0 pixels to render), but will still build; while the
    // second renderFrame call below won't do any build (already done)
    // but do all the rendering (Nx*Ny pixels) ...
    auto t0 = std::chrono::system_clock::now();
    renderFrame(0,0);
    auto t1 = std::chrono::system_clock::now();
    std::cout << "done building optix data structures, which took "
              << std::setprecision(4) << std::chrono::duration<double>(t1-t0).count()
              << " seconds" << std::endl;
  }
#endif

  // render the frame (and time it)
  auto t0 = std::chrono::system_clock::now();
  renderFrame(Nx, Ny);
  auto t1 = std::chrono::system_clock::now();
  std::cout << "done rendering, which took "
            << std::setprecision(4) << std::chrono::duration<double>(t1-t0).count()
            << " seconds (for " << numSamples << " paths per pixel)" << std::endl;
       
  // ... map it, save it, and cleanly unmap it after reading...
  const vec3f *pixels = (const vec3f *)fb->map();
  savePPM("finalChapter.ppm",Nx,Ny,pixels);
  fb->unmap();

  // ... done.
  return 0;
}

