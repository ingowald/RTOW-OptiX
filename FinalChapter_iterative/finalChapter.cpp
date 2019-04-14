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
#include <optixu/optixu_matrix_namespace.h>
//yaml (for parsing config file)
#include <yaml-cpp/yaml.h>
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
#include <string>
#include <vector>
#include <limits>

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
optix::Transform createSphereXform(const vec3f &center, const optix::Matrix3x3& upperLeft, const float radius, const optix::GeometryGroup &gg)
{
  //create transform based on the given center and radius
  /*
   *float sphereMatRaw[16] =
   *{                    
   *  radius, 0.0f,   0.0f,   center.x,
   *  0.0f,   radius, 0.0f,   center.y, 
   *  0.0f,   0.0f,   radius, center.z,
   *  0.0f,   0.0f,   0.0f,   1.0f                                 
   *};                                                              
   */
  float sphereMatRaw[16] =
  {                    
    radius*upperLeft[0], radius*upperLeft[1], radius*upperLeft[2], center.x,
    radius*upperLeft[3], radius*upperLeft[4], radius*upperLeft[5], center.y, 
    radius*upperLeft[6], radius*upperLeft[7], radius*upperLeft[8], center.z,
    0.0f,   0.0f,   0.0f,   1.0f                                 
  };                                                              
  optix::Matrix4x4 matrixSphere(sphereMatRaw);   
  optix::Transform trSphere = g_context->createTransform();
  trSphere->setMatrix(false, matrixSphere.getData(),
      matrixSphere.inverse().getData());    
  
  trSphere->setChild(gg); 

  return trSphere;
}

optix::Group createScene(const std::string& filename)
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

  // --------------Uplift this code later to main()-------
  std::string line;
  //std::ifstream csvfile("../tensor.csv");
  std::ifstream csvfile(filename);
  int count =0; // This is just to limit the amount of the file we read for testing
  if(csvfile.is_open()){
	  while(getline(csvfile,line)){
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

        //assuming row major order
        optix::Matrix3x3 tensor;
        tensor = optix::Matrix3x3::identity();

        for (int i=0; i < 9; i++){
          tensor[i] = row[i];
          //printf("%f %f %f\n%f %f %f\n%f %f %f\n",row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]);
          //std::cout << " ********* " << std::endl;
        }

        optix::Matrix3x3 symmetrized_tensor = 0.5f*(tensor + tensor.transpose());

				//t_list.push_back(createSphereXform(center, symmetrized_tensor, 0.001f, ggDiffuse));
        t_list.push_back(createSphereXform(center, symmetrized_tensor, 0.2f * 0.001592912349527057f, ggDiffuse));
		  }
		  count++;
	  }
	  csvfile.close();
  }
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

int main(int argc, char **argv)
{
  if(argc != 2){ 
    std::cout << "Usage: ./finalChapter_iterative <config_file>.yaml" << std::endl;
    exit(1);
  }

  //Initialize floating-point valuse to NaNs instead of leaving them
  //uninitialized. Since NaN's propagate, this makes the program easier to
  //debug if we are using an uninitialized value somewhere we shouldn't.
  //const float nan = std::numeric_limits<float>::quiet_NaN();
  //float fovy = nan; 
  //vec3f cam_pos(NAN);

  std::string config_file_name = std::string(argv[1]);
  size_t config_fname_len = config_file_name.length();
  std::string ext(".yaml");
  if( config_fname_len < ext.length() || config_file_name.compare(config_fname_len - ext.length(),ext.length(), ext) != 0 ){
    std::cout << "Input must be a .yaml file!" << std::endl;
    exit(1);
  }

  std::cout << "Loading config, in YAML format, at: " << config_file_name << std::endl;
  YAML::Node config = YAML::LoadFile(config_file_name);
  const float fovy = config["camera"]["fovy"].as<float>(); 

  YAML::Node cam_pos = config["camera"]["position"];
  assert(cam_pos.IsSequence());
  const vec3f lookfrom(cam_pos[0].as<float>(), cam_pos[1].as<float>(), cam_pos[2].as<float>());

  YAML::Node lookat_pos = config["camera"]["lookat"];
  assert(lookat_pos.IsSequence());
  const vec3f lookat(lookat_pos[0].as<float>(), lookat_pos[1].as<float>(), lookat_pos[2].as<float>());

  YAML::Node up_dir = config["camera"]["up"];
  assert(up_dir.IsSequence());
  const vec3f up(up_dir[0].as<float>(), up_dir[1].as<float>(), up_dir[2].as<float>());

  const std::string data_file = config["data_file"].as<std::string>();

  std::cout << "Camera position: " << lookfrom.x << " " << lookfrom.y << " " << lookfrom.z << " " << std::endl;
  std::cout << "Lookat position:" << lookat.x << " " << lookat.y << " " << lookat.z << " " << std::endl;
  std::cout << "Up direction:" << up.x << " " << up.y << " " << up.z << " " << std::endl;
  std::cout << "Field of view (y), degrees: " << fovy << std::endl;
  std::cout << "Data file: " << data_file << std::endl;
  
  //exit(0);

  if(argc != 2){ //OLD CODE! 
    std::cout << "Usage: ./finalChapter_iterative <data_file>.csv" << std::endl;
    exit(1);
  }

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

  //const vec3f lookfrom(1.1487395261676667,-0.324271485442182,1.0268790810616117);
  //const vec3f lookat(-42.92895065482805,-32.67156564843721,-18.723725570334892);
  Camera camera(lookfrom,
                lookat,
                /* up */ vec3f(0.6437328837528422, -0.7215645820362587, -0.2548578590628296 ),//up,
                /* fovy, in degrees */ 30, //fovy, 
                /* aspect */ float(Nx) / float(Ny),
                /* aperture */ 0.01f, //0.01f
                /* dist to focus: */ 1.0f);
  camera.set();

  // set the ray generation and miss shader program
  setRayGenProgram();
  setMissProgram();

  // create a frame buffer
  optix::Buffer fb = createFrameBuffer(Nx, Ny);
  g_context["fb"]->set(fb);

  // create the world to render
  //optix::GeometryGroup world = createScene();
  //optix::Group world = createScene(std::string(argv[1]));
  optix::Group world = createScene(data_file); //FIXME: This is a HACK!
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

