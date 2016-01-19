#include <v4r/rendering/depthmapRenderer.h>

#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp>
#include <GL/gl.h>

#include <X11/X.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);
static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
static glXMakeContextCurrentARBProc glXMakeContextCurrentARB = 0;

int counter=0;
Display* dpy;
GLXContext ctx;

namespace v4r
{

bool DepthmapRenderer::glfwRunning=false;


int DepthmapRenderer::search_midpoint(int &index_start, int &index_end, size_t &n_vertices, int &edge_walk,
    std::vector<int> &midpoint, std::vector<int> &start, std::vector<int> &end, std::vector<float> &vertices)
{
  for (int i = 0; i < edge_walk; i++)
    if ((start[i] == index_start && end[i] == index_end) || (start[i] == index_end && end[i] == index_start)) {
      int res_tmp = midpoint[i];

      /* update the arrays */
      start[i] = start[edge_walk - 1];
      end[i] = end[edge_walk - 1];
      midpoint[i] = midpoint[edge_walk - 1];
      edge_walk--;

      return res_tmp;
    }

  /* vertex not in the list, so we add it */
  start[edge_walk] = index_start;
  end[edge_walk] = index_end;
  midpoint[edge_walk] = n_vertices;

  /* create new vertex */
  vertices[3 * n_vertices] = (vertices[3 * index_start] + vertices[3 * index_end]) / 2.0f;
  vertices[3 * n_vertices + 1] = (vertices[3 * index_start + 1] + vertices[3 * index_end + 1]) / 2.0f;
  vertices[3 * n_vertices + 2] = (vertices[3 * index_start + 2] + vertices[3 * index_end + 2]) / 2.0f;

  /* normalize the new vertex */
  float length = sqrt(
      vertices[3 * n_vertices] * vertices[3 * n_vertices] + vertices[3 * n_vertices + 1] * vertices[3 * n_vertices + 1]
          + vertices[3 * n_vertices + 2] * vertices[3 * n_vertices + 2]);
  length = 1 / length;
  vertices[3 * n_vertices] *= length;
  vertices[3 * n_vertices + 1] *= length;
  vertices[3 * n_vertices + 2] *= length;

  n_vertices++;
  edge_walk++;
  return midpoint[edge_walk - 1];
}

void DepthmapRenderer::subdivide(size_t &n_vertices, size_t &n_edges, size_t &n_faces, std::vector<float> &vertices, std::vector<int> &faces)
{
    //Code i stole from thomas moerwald:
    int n_vertices_new = n_vertices + 2 * n_edges;
    int n_faces_new = 4 * n_faces;

    int edge_walk = 0;
    n_edges = 2 * n_vertices + 3 * n_faces;

    std::vector<int> start(n_edges);
    std::vector<int> end(n_edges);
    std::vector<int> midpoint(n_edges);

    std::vector<int> faces_old = faces;
    vertices.resize(3 * n_vertices_new);
    faces.resize(3 * n_faces_new);
    n_faces_new = 0;

    for (size_t i = 0; i < n_faces; i++) {
      int a = faces_old[3 * i];
      int b = faces_old[3 * i + 1];
      int c = faces_old[3 * i + 2];

      int ab_midpoint = search_midpoint(b, a, n_vertices, edge_walk, midpoint, start, end, vertices);
      int bc_midpoint = search_midpoint(c, b, n_vertices, edge_walk, midpoint, start, end, vertices);
      int ca_midpoint = search_midpoint(a, c, n_vertices, edge_walk, midpoint, start, end, vertices);

      faces[3 * n_faces_new] = a;
      faces[3 * n_faces_new + 1] = ab_midpoint;
      faces[3 * n_faces_new + 2] = ca_midpoint;
      n_faces_new++;
      faces[3 * n_faces_new] = ca_midpoint;
      faces[3 * n_faces_new + 1] = ab_midpoint;
      faces[3 * n_faces_new + 2] = bc_midpoint;
      n_faces_new++;
      faces[3 * n_faces_new] = ca_midpoint;
      faces[3 * n_faces_new + 1] = bc_midpoint;
      faces[3 * n_faces_new + 2] = c;
      n_faces_new++;
      faces[3 * n_faces_new] = ab_midpoint;
      faces[3 * n_faces_new + 1] = b;
      faces[3 * n_faces_new + 2] = bc_midpoint;
      n_faces_new++;
    }
    n_faces = n_faces_new;
}

DepthmapRenderer::DepthmapRenderer(int resx, int resy)
{
    //First of all: create opengl context:
    //res=glm::ivec2(resx,resy);
    res=Eigen::Vector2i(resx,resy);

    if(counter==0){
        //BEGIN OF COPYCAT CODE
        static int visual_attribs[] = {
                None
        };
        int context_attribs[] = {
                GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
                GLX_CONTEXT_MINOR_VERSION_ARB, 0,
                None
        };

        dpy = XOpenDisplay(0);
        int fbcount = 0;
        GLXFBConfig* fbc = NULL;

        GLXPbuffer pbuf;

        /* open display */
        if ( ! (dpy = XOpenDisplay(0)) ){
                fprintf(stderr, "Failed to open display\n");
                exit(1);
        }

        /* get framebuffer configs, any is usable (might want to add proper attribs) */
        if ( !(fbc = glXChooseFBConfig(dpy, DefaultScreen(dpy), visual_attribs, &fbcount) ) ){
                fprintf(stderr, "Failed to get FBConfig\n");
                exit(1);
        }

        /* get the required extensions */
        glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB");
        glXMakeContextCurrentARB = (glXMakeContextCurrentARBProc)glXGetProcAddressARB( (const GLubyte *) "glXMakeContextCurrent");
        if ( !(glXCreateContextAttribsARB && glXMakeContextCurrentARB) ){
                fprintf(stderr, "missing support for GLX_ARB_create_context\n");
                XFree(fbc);
                exit(1);
        }

        /* create a context using glXCreateContextAttribsARB */
        if ( !( ctx = glXCreateContextAttribsARB(dpy, fbc[0], 0, True, context_attribs)) ){
                fprintf(stderr, "Failed to create opengl context\n");
                XFree(fbc);
                exit(1);
        }

        /* create temporary pbuffer */
        int pbuffer_attribs[] = {
                GLX_PBUFFER_WIDTH, 800,
                GLX_PBUFFER_HEIGHT, 600,
                None
        };
        pbuf = glXCreatePbuffer(dpy, fbc[0], pbuffer_attribs);

        XFree(fbc);
        XSync(dpy, False);

        /* try to make it the current context */
        if ( !glXMakeContextCurrent(dpy, pbuf, pbuf, ctx) ){
            /* some drivers does not support context without default framebuffer, so fallback on
             * using the default window.
             */
            if ( !glXMakeContextCurrent(dpy, DefaultRootWindow(dpy), DefaultRootWindow(dpy), ctx) ){
                fprintf(stderr, "failed to make current\n");
                exit(1);
            }
        }


        /* try it out */
        // printf("vendor: %s\n", (const char*)glGetString(GL_VENDOR));
        //END OF COPYCATCODE
    }
    counter++;


    GLenum err=glewInit();

    if(err!=GLEW_OK){
        std::stringstream s; s << "glewInit failed, aborting. " << err;
        throw std::runtime_error(s.str());
    }

    glGetError();
    //create framebuffer:

    //Hardcoded shader:
    const char *vertex=
            "#version 450 \n\
            in vec4 pos;\n\
            out vec4 colorIn;\n\
            void main(){\n\
               gl_Position=vec4(pos.xyz,1);\n\
               colorIn=unpackUnorm4x8(floatBitsToUint(pos.w));\n\
            }";
    const char *geometry=
            "#version 450\n\
            \
            layout(binding=0, offset=0) uniform atomic_uint faceCount;\n\
            //layout(binding=0, offset=4) uniform atomic_uint backFacing;\n\
            layout(std430,binding=1) buffer Buffer{\n\
                vec2 AnPixCnt[];\n\
            };\n\
            layout (triangles) in;//triangles\n\
            layout(triangle_strip,max_vertices=3) out;\n\
            noperspective out float z;\n\
            in vec4 colorIn[];\n\
            flat out unsigned int index;\n\
            out vec4 color;\n\
            uniform vec4 projection;\n\
            uniform mat4 transformation;\n\
            uniform ivec2 viewportRes;\n\
            vec4 project(vec4 pos){\n\
                return vec4((pos.x*projection.x/pos.z+projection.z)*2.0-1.0,(pos.y*projection.y/pos.z+projection.w)*2.0-1.0,0.1/pos.z,1);//max draw distance is 10 m\n\
            }\n\
            void main(){\
                unsigned int ind = atomicCounterIncrement(faceCount);\n\
                index= ind+1;\n\
                gl_Position=project(transformation*gl_in[0].gl_Position);\n\
                vec4 p1=transformation*gl_in[0].gl_Position;\n\
                vec2 pp1=gl_Position.xy;\n\
                z=-(transformation*gl_in[0].gl_Position).z;\n\
                color=colorIn[0];\n\
                EmitVertex();\n\
                gl_Position=project(transformation*gl_in[1].gl_Position);\n\
                vec2 pp2=gl_Position.xy;\n\
                vec4 p2=transformation*gl_in[1].gl_Position;\n\
                z=-(transformation*gl_in[1].gl_Position).z;\n\
                color=colorIn[1];\n\
                EmitVertex();\n\
                gl_Position=project(transformation*gl_in[2].gl_Position);\n\
                vec4 p3=transformation*gl_in[2].gl_Position;\n\
                vec2 pp3=gl_Position.xy;\n\
                z=-(transformation*gl_in[2].gl_Position).z;\n\
                color=colorIn[2];\n\
                EmitVertex();\n\
                //calc triangle surface area\n\
                float A= length(cross(vec3(p1)/p1.w-vec3(p3)/p3.w,vec3(p2)/p2.w-vec3(p3)/p3.w));//TODO: Change this to correct pixel area calculation\n\
                vec3 a=vec3((pp2.x-pp1.x)*float(viewportRes.x),(pp2.y-pp1.y)*float(viewportRes.y),0)*0.5;\n\
                vec3 b=vec3((pp3.x-pp1.x)*float(viewportRes.x),(pp3.y-pp1.y)*float(viewportRes.y),0)*0.5;\n\
                float Apix=length(cross(a,b))*0.5;\n\
                AnPixCnt[ind]=vec2(A,Apix);\n\
            }";
    const char *fragment=
            "#version 450 \n\
            \
            noperspective in float z;\n\
            flat in unsigned int index;\n\
            in vec4 color;\n\
            out vec4 depthOutput;\n\
            out vec4 colorOutput;\n\
            out unsigned int indexOutput;\n\
            \n\
            uniform float zNear;\n\
            uniform float zFar;\n\
            void main(){\n\
            \
               depthOutput=vec4(z,z,z,1);\n\
               depthOutput=vec4(vec3(-0.1/((gl_FragCoord.z/gl_FragCoord.w)*2.0-1.0)),1);\n\
               //depthOutput=vec4(0.1,z,z,1);\n\
               indexOutput=index;//uintBitsToFloat(1234);\n\
               colorOutput=color;\n\
            }";


    //Compile and link shader:
    //std::cout << "compiling vertex shader:" << std::endl;
    GLuint vertexShader =glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader,1,(const GLchar**)&vertex,NULL);
    glCompileShader(vertexShader);
    GLint status;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
    char buffer[512];
    glGetShaderInfoLog(vertexShader, 512, NULL, buffer);
    //std::cout << buffer << std::endl;

    //std::cout << "compiling geometry shader:" << std::endl;
    GLuint geometryShader =glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(geometryShader,1,(const GLchar**)&geometry,NULL);
    glCompileShader(geometryShader);
    glGetShaderiv(geometryShader, GL_COMPILE_STATUS, &status);
    glGetShaderInfoLog(geometryShader, 512, NULL, buffer);
    //std::cout << buffer << std::endl;


    //std::cout << "compiling fragment shader:" << std::endl;
    GLuint fragmentShader =glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader,1,(const GLchar**)&fragment,NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
    glGetShaderInfoLog(fragmentShader, 512, NULL, buffer);
    //std::cout << buffer << std::endl;

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, geometryShader);
    glAttachShader(shaderProgram, fragmentShader);

    glBindFragDataLocation(shaderProgram, 0, "depthOutput");
    glBindFragDataLocation(shaderProgram, 1, "indexOutput");
    glBindFragDataLocation(shaderProgram, 2, "colorOutput");
    glLinkProgram(shaderProgram);

    glGetProgramiv( shaderProgram, GL_LINK_STATUS, &status);
    if( GL_FALSE == status ) {
        // Store log and return false
        int length = 0;
        std::string logString;

        glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &length );

        if( length > 0 ) {
            char * c_log = new char[length];
            int written = 0;
            glGetProgramInfoLog(shaderProgram, length, &written, c_log);
            logString = c_log;
            std::cout << logString << std::endl;
            delete [] c_log;
        }
    }

    //Get uniform locations:
    projectionUniform=glGetUniformLocation(shaderProgram,"projection");
    poseUniform = glGetUniformLocation(shaderProgram,"transformation");
    viewportResUniform = glGetUniformLocation(shaderProgram,"viewportRes");

    //get attribute location
    posAttribute=glGetAttribLocation(shaderProgram,"pos");


    //generate framebuffer:
    glGenFramebuffers(1,&FBO);
    glBindFramebuffer(GL_FRAMEBUFFER,FBO);

    //depth texture:
    glGenTextures(1,&depthTex);
    glBindTexture(GL_TEXTURE_2D,depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_R32F, resx, resy, 0,GL_RED, GL_FLOAT, 0);

    //index texture:
    glGenTextures(1,&indexTex);
    glBindTexture(GL_TEXTURE_2D,indexTex);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_R32UI, resx, resy, 0,GL_RED_INTEGER, GL_UNSIGNED_INT, 0);

    //color texture:
    glGenTextures(1,&colorTex);
    glBindTexture(GL_TEXTURE_2D,colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA8, resx, resy, 0,GL_RGBA, GL_UNSIGNED_BYTE, 0);

    //z buffer
    glGenRenderbuffers(1, &zBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, zBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, resx, resy);//GL_DEPTH_COMPONENT_32F //without the 32F?
    //glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,zBuffer);
    GLuint buffers[]={GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2};//last one is for debug
    glDrawBuffers(3,buffers);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,depthTex,0);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,indexTex,0);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D,colorTex,0);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
        std::cout << "Framebuffer not complete" << std::endl;
    }
    glViewport(0,0,resx,resy);


    //two atomic counters to give each triangle a specific index
    glGenBuffers(1, &atomicCounterBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint)*2, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicCounterBuffer);


    //shader storage buffer for storing surface area and theoretical pixel area for each pixel:
    glGenBuffers(1,&SSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec2)*maxMeshSize, 0, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, SSBO);

    glGenVertexArrays(1,&VAO);

    if( (err = glGetError()) != GL_NO_ERROR)
        std::cerr << "An OpenGL error occured during initialization (" << err << ")" << std::endl;

    //There is no geometry defined yet!
    VBO=0;
    IBO=0;

}

DepthmapRenderer::~DepthmapRenderer()
{
    //delete the framebuffer:
    glDeleteTextures(1,&depthTex);
    glDeleteTextures(1,&indexTex);
    glDeleteTextures(1,&colorTex);
    glDeleteBuffers(1,&zBuffer);
    glDeleteFramebuffers(1,&FBO);

    //delete the shader:
    glDeleteShader(shaderProgram);

    //delete the VAO:
    glDeleteVertexArrays(1,&VAO);
    if(VBO){
        glDeleteBuffers(1,&VBO);
        glDeleteBuffers(1,&IBO);
    }

    //remaining buffers
    glDeleteBuffers(1,&SSBO);
    glDeleteBuffers(1,&atomicCounterBuffer);


    counter--;
    //once every renderer closes destroy the context:
    if(counter==0){
        glXDestroyContext(dpy,ctx);
    }

}

std::vector<Eigen::Vector3f> DepthmapRenderer::createSphere(float r, size_t subdivisions)
{

    std::vector<Eigen::Vector3f> result;

    float t = (1 + sqrt(5.0f)) / 2;
    float tau = t / sqrt(1 + t * t);
    float one = 1 / sqrt(1 + t * t);

    float icosahedron_vertices[] = { tau, one, 0.0, -tau, one, 0.0, -tau, -one, 0.0, tau, -one, 0.0, one, 0.0, tau, one, 0.0, -tau,
        -one, 0.0, -tau, -one, 0.0, tau, 0.0, tau, one, 0.0, -tau, one, 0.0, -tau, -one, 0.0, tau, -one };
    int icosahedron_faces[] = { 4, 8, 7, 4, 7, 9, 5, 6, 11, 5, 10, 6, 0, 4, 3, 0, 3, 5, 2, 7, 1, 2, 1, 6, 8, 0, 11, 8, 11, 1, 9,
        10, 3, 9, 2, 10, 8, 4, 0, 11, 0, 5, 4, 9, 3, 5, 3, 10, 7, 8, 1, 6, 1, 11, 7, 2, 9, 6, 10, 2 };

    size_t n_vertices = 12;
    size_t n_faces = 20;
    size_t n_edges = 30;


    std::vector<float> vertices(3 * n_vertices);
    std::vector<int> faces(3 * n_faces);

    for (size_t i = 0; i < (3 * n_vertices); i++)
      vertices[i]= icosahedron_vertices[i];

    for (size_t i = 0; i < (3 * n_faces); i++)
      faces[i] = icosahedron_faces[i];



    for (size_t i = 0; i < subdivisions; i++)
      subdivide(n_vertices, n_edges, n_faces, vertices, faces);

    // Copy vertices
    for (size_t i = 0; i < n_vertices; i++) {
        Eigen::Vector3f v;
        v[0] = r * vertices[3 * i + 0];
        v[1] = r * vertices[3 * i + 1];
        v[2] = r * vertices[3 * i + 2];
        result.push_back(v);
    }

    return result;

}

void DepthmapRenderer::setIntrinsics(float fx, float fy, float cx, float cy)
{
    fxycxy=Eigen::Vector4f(fx,fy,cx,cy);
}

void DepthmapRenderer::setModel(DepthmapRendererModel *_model)
{
    this->model=_model;

    //bind shader:
    glBindVertexArray(VAO);
    glUseProgram(shaderProgram);
    this->model->loadToGPU(VBO,IBO);

    //set vertexAttribArray
    glEnableVertexAttribArray(posAttribute);
    glVertexAttribPointer(posAttribute,4,GL_FLOAT,GL_FALSE,sizeof(glm::vec4),0);
    glBindVertexArray(0);
    //maybe upload it to
}

Eigen::Matrix4f DepthmapRenderer::getPoseLookingToCenterFrom(Eigen::Vector3f position)
{
    glm::vec3 up(0,0,1);
    if(position[0]==0 && position[1]==0){
        up=glm::vec3(1,0,0);
    }
    glm::vec3 pos(position[0],position[1],position[2]);
    glm::vec3 center(0,0,0);

    glm::mat4 pose_tmp = glm::lookAt(pos,center,up);

    //transform to Eigen Matrix type
    Eigen::Matrix4f ePose;
    for(size_t i=0; i<4; i++){
        for(size_t j=0; j<4; j++){
            ePose(i,j) = pose_tmp[i][j];
        }
    }
    return ePose;
}

void DepthmapRenderer::setCamPose(Eigen::Matrix4f _pose)
{
    this->pose=_pose;
}



cv::Mat DepthmapRenderer::renderDepthmap(float &visible,cv::Mat &color) const
{
    //load shader:
    glUseProgram(shaderProgram);

    //set the uniforms
    glm::mat4 gPose; //Keep this conversion code... might be useful
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            gPose[i][j]=pose(i,j);
        }
    }

    glUniformMatrix4fv(poseUniform,1,GL_FALSE,(float*)&gPose);
    glUniform4f(projectionUniform,fxycxy[0]/(float)res[0],fxycxy[1]/(float)res[1],fxycxy[2]/(float)res[0],fxycxy[3]/(float)res[1]);
    glUniform2i(viewportResUniform,res[0],res[1]);

    //use vertex array object:
    glBindVertexArray(VAO);

    //activate fbo
    glBindFramebuffer(GL_FRAMEBUFFER,FBO);

    //set viewport and clear buffers
    glViewport(0,0,res[0],res[1]);
    glClearColor(0.0,0.0,0.0,1);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,  SSBO);

    //setup the atomic variable for counting the triangles
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    GLuint* ptr = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint),
                                            GL_MAP_WRITE_BIT |
                                            GL_MAP_INVALIDATE_BUFFER_BIT |
                                            GL_MAP_UNSYNCHRONIZED_BIT);
    ptr[0] = 0;
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);


    //disable culling (Off by default but this would be the right moment)
    //render
    glDrawElements(
        GL_TRIANGLES,      // mode
        model->getIndexCount(),    // count
        GL_UNSIGNED_INT,   // type
        (void*)0           // element array buffer offset
    );

    glFinish();

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);//not helping either
    //read buffers:
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounterBuffer);
    ptr = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint)*2,
                                            GL_MAP_READ_BIT );

    //std::cout << ptr[0] << " visible triangles and " << ptr[1] << " invisible triangles" << std::endl;
    size_t faceCount = ptr[0];
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    //std::cout << "face count " << faceCount << std::endl;
    //download fbo

    //GET DEPTH TEXTURE
    cv::Mat depthmap(res[1],res[0],CV_32FC1);//FC1
    glBindTexture(GL_TEXTURE_2D,depthTex);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RED,GL_FLOAT,depthmap.data);
    //glGetTexImage(GL_TEXTURE_2D,0,GL_RED,GL_FLOAT,depthmap.data);



    //GET INDEX TEXTURE
    cv::Mat indexMap(res[1],res[0],CV_32SC1);
    glBindTexture(GL_TEXTURE_2D,indexTex);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RED_INTEGER,GL_UNSIGNED_INT,indexMap.data);


    //GET SSBO DATA
    glm::vec2* faceSurfaceArea=new glm::vec2[faceCount];
    glBindBuffer(GL_ARRAY_BUFFER,SSBO);//GL_SHADER_STORAGE_BUFFER
    glGetBufferSubData(GL_ARRAY_BUFFER,0,sizeof(glm::vec2)*faceCount,faceSurfaceArea);

    //GET COLOR TEXTURE
    cv::Mat colorMat(res[1],res[0],CV_8UC4);
    glBindTexture(GL_TEXTURE_2D,colorTex);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,colorMat.data);
    //imshow("colorMat",colorMat);
    color=colorMat;

    //get pixel count for every triangle
    int* facePixelCount=new int[faceCount]();//hopefully initzialized with zero
    for(int u=0;u<depthmap.rows;u++){
        for(int v=0;v<depthmap.cols;v++){
            if(indexMap.at<int>(u,v)!=0){
                facePixelCount[indexMap.at<int>(u,v)-1]++;
            }
        }
    }
    float visibleArea=0;
    float fullArea=0;
    int fullPixelCount=0;
    //Sum up the full surface area and the visible surface area
    for(size_t i=0;i<faceCount;i++){
        //std::cout << "pixel count face " << i << ": " << facePixelCount[i]<< std::endl;
        fullPixelCount+=facePixelCount[i];
        fullArea+=faceSurfaceArea[i].x;
        float pixelForFace=faceSurfaceArea[i].y;
        if(pixelForFace!=0){
            visibleArea+=faceSurfaceArea[i].x*float(facePixelCount[i])/pixelForFace;
        }
    }
    //calc
    visible=visibleArea/fullArea;


    GLuint err;
    if( (err = glGetError()) != GL_NO_ERROR)
        std::cerr << "A terrible OpenGL error occured during rendering (" << err << ")" << std::endl;

    delete[] facePixelCount;
    delete[] faceSurfaceArea;
    return depthmap;
}

pcl::PointCloud<pcl::PointXYZ> DepthmapRenderer::renderPointcloud(float &visibleSurfaceArea) const
{
    const float bad_point = std::numeric_limits<float>::quiet_NaN();
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width    = res[0];
    cloud.height   = res[1];
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

    //set pose inside pcl structure
    /*Eigen::Matrix4f ePose;
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            ePose(i,j)=pose[i][j];
        }
    }*/
    cloud.sensor_orientation_ = Eigen::Quaternionf(Eigen::Matrix3f(pose.block(0,0,3,3)));
    Eigen::Vector3f trans = Eigen::Matrix3f(pose.block(0,0,3,3))*Eigen::Vector3f(pose(3,0),pose(3,1),pose(3,2));
    cloud.sensor_origin_ = Eigen::Vector4f(trans(0), trans(1), trans(2), 1.0f);

    cv::Mat color;
    cv::Mat depth=renderDepthmap(visibleSurfaceArea,color);
    for(size_t k=0;k<cloud.height;k++){
        for(size_t j=0;j<cloud.width;j++){
            float d=depth.at<float>(k,j);
            if(d==0){
                cloud.at(j,k)=pcl::PointXYZ(bad_point,bad_point,bad_point);
            }
            else{
                pcl::PointXYZ p;
                p.x=((float)j-fxycxy[2])/fxycxy[0]*d;
                p.y=((float)k-fxycxy[3])/fxycxy[1]*d;
                p.z=d;

                cloud.at(j,k)=p;
            }
        }
    }


    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> DepthmapRenderer::renderPointcloudColor(float &visibleSurfaceArea) const
{
    const float bad_point = std::numeric_limits<float>::quiet_NaN();
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.width    = res[0];
    cloud.height   = res[1];
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

    //set pose inside pcl structure
    /*Eigen::Matrix4f ePose;
    for(size_t i=0; i<4; i++){
        for(size_t j=0; j<4; j++){
            ePose(i,j)=pose[i][j];
        }
    }*/
    cloud.sensor_orientation_ = Eigen::Quaternionf(Eigen::Matrix3f(pose.block(0,0,3,3)));
    Eigen::Vector3f trans = Eigen::Matrix3f(pose.block(0,0,3,3))*Eigen::Vector3f(pose(3,0),pose(3,1),pose(3,2));
    cloud.sensor_origin_ = Eigen::Vector4f(trans(0),trans(1),trans(2),1.0f);

    cv::Mat color;
    const cv::Mat depth = renderDepthmap(visibleSurfaceArea,color);
    cv::vector<cv::Mat> color_channels(3);
    cv::split(color, color_channels);
    cv::Mat b, g, r;
    b = color_channels[0];
    g = color_channels[1];
    r = color_channels[2];

    for(size_t k=0;k<cloud.height;k++){
        for(size_t j=0;j<cloud.width;j++){
            float d=depth.at<float>(k,j);
            if(d==0){
                cloud.at(j,k).x=bad_point;
                cloud.at(j,k).y=bad_point;
                cloud.at(j,k).z=bad_point;
            }
            else{
                pcl::PointXYZRGB p;
                p.x=((float)j-fxycxy[2])/fxycxy[0]*d;
                p.y=((float)k-fxycxy[3])/fxycxy[1]*d;
                p.z=d;
                p.r = r.at<unsigned char>(k,j);
                p.g = g.at<unsigned char>(k,j);
                p.b = b.at<unsigned char>(k,j);
                cloud.at(j,k)=p;
            }

        }
    }

    return cloud;
}

}
