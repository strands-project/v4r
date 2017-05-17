#include <v4r/rendering/dmRenderObject.h>
#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

///TODO: remove every trace of glm
#include <glm/glm.hpp>

#include <GL/glew.h>
#include <GL/gl.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/Vertices.h>
#include <pcl/ros/conversions.h>

namespace v4r
{
struct DepthmapRendererModel::Vertex{
    glm::vec3 pos;
    glm::u8vec4 rgba;
};

DepthmapRendererModel::DepthmapRendererModel(const std::string &file, bool shiftToCenterAndNormalizeScale)
{
    vertexCount=0;
    indexCount=0;
    geometry=false;

    color=false;
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile( file,
            aiProcess_CalcTangentSpace       |
            aiProcess_Triangulate            |
            aiProcess_JoinIdenticalVertices  |
            aiProcess_SortByPType);

    if( !scene)
    {
      std::cout << importer.GetErrorString() << std::endl;
      return;
    }

    geometry=true;


    glm::dvec3 mean(0,0,0);
    //first count how many geometry we need + find out maxima and average...
    if(scene->HasMeshes()){
        for(size_t i=0;i<scene->mNumMeshes;i++){

            vertexCount+=scene->mMeshes[i]->mNumVertices;

            for(size_t j=0;j<scene->mMeshes[i]->mNumFaces;j++){
                indexCount += scene->mMeshes[i]->mFaces[i].mNumIndices;
            }
            //std::cout << "Mesh nr:" << i << " faces:"<< scene->mMeshes[i]->mNumFaces << "  Vertices:"<< scene->mMeshes[i]->mNumVertices << std::endl;
            for(size_t j=0;j<scene->mMeshes[i]->mNumVertices;j++){
                //print out vertex data:
                glm::vec3 vertex(scene->mMeshes[i]->mVertices[j].x,scene->mMeshes[i]->mVertices[j].y,scene->mMeshes[i]->mVertices[j].z);
                //std::cout << " vertex:" <<vertex.x << " " << vertex.y << " " << vertex.z << std::endl;//Debug
                mean+=glm::dvec3(vertex.x,vertex.y,vertex.z);
                color=color||scene->mMeshes[i]->HasVertexColors(0);
            }
        }
    }
    mean=mean/((double)vertexCount);
    //last: create memory for geometry
    indices=new uint32_t[indexCount];
    vertices=new Vertex[vertexCount];
    double maxDistToCenter=0.0;

    unsigned int k=0;
    unsigned int l=0;
    unsigned int m=0;
    for(size_t i=0;i<scene->mNumMeshes;i++){
        for(size_t j=0;j<scene->mMeshes[i]->mNumVertices;j++){
            Vertex v;
            v.pos=glm::vec3(scene->mMeshes[i]->mVertices[j].x,
                            scene->mMeshes[i]->mVertices[j].y,
                            scene->mMeshes[i]->mVertices[j].z);
            if(shiftToCenterAndNormalizeScale){
                v.pos=v.pos-glm::vec3(mean);
            }
            if(scene->mMeshes[i]->HasVertexColors(0)){
                aiColor4D* colors= scene->mMeshes[i]->mColors[0];
                v.rgba=glm::u8vec4(colors[j].r*255.0f,colors[j].g*255.0f,colors[j].b*255.0f,colors[j].a*255.0f);
                //v.rgba=glm::u8vec4(255,255,255,255);//debug
            }else{
                v.rgba=glm::u8vec4();
            }
            vertices[k]=v;
            float distToCenter=glm::length(vertices[k].pos);
            if(!shiftToCenterAndNormalizeScale){
                distToCenter=glm::length(vertices[k].pos-glm::vec3(mean));
            }
            if(distToCenter>maxDistToCenter){
                maxDistToCenter=distToCenter;
            }
            k++;
        }

        for(size_t j=0;j<scene->mMeshes[i]->mNumFaces;j++){
            for(size_t n=0;n<scene->mMeshes[i]->mFaces[j].mNumIndices;n++){

                indices[m]=scene->mMeshes[i]->mFaces[j].mIndices[n]+l;
                m++;
            }
        }
        l+=scene->mMeshes[i]->mNumVertices;
    }

    offset=Eigen::Vector3f(-mean.x,-mean.y,-mean.z);

    scale=1.0f/(float)maxDistToCenter;

    //scale it:
    if(shiftToCenterAndNormalizeScale){
        for(uint32_t i=0;i<vertexCount;i++){
            vertices[i].pos=vertices[i].pos*scale;
            //std::cout << " vertex:" <<vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << std::endl;//Debug
        }
    }
}

DepthmapRendererModel::DepthmapRendererModel(const pcl::PolygonMesh& pclMesh, bool shiftToCenterAndNormalizeScale){

    pcl::PointCloud<pcl::PointXYZ> points;
    pcl::fromPCLPointCloud2(pclMesh.cloud, points);


    color=false;
    if(points.size()>3){
        geometry=true;
    }


    vertexCount=points.size();
    vertices=new Vertex[vertexCount];


    glm::dvec3 mean(0,0,0);
    for(uint32_t i=0;i<vertexCount;i++){
        Vertex vert;
        pcl::PointXYZ point = points.at(i);
        vert.pos=glm::vec3(point.x,point.y,point.z);
        vert.rgba=glm::u8vec4(128,128,128,255);
        vertices[i]=vert;
        mean+=glm::dvec3(point.x,point.y,point.z);

    }
    mean=mean/(double)vertexCount;

    float maxDistToCenter=0.0;
    for(uint32_t i=0;i<vertexCount;i++){

        float distToCenter=glm::length(vertices[i].pos-glm::vec3(mean));
        if(distToCenter>maxDistToCenter){
            maxDistToCenter=distToCenter;
        }
        if(shiftToCenterAndNormalizeScale){
            vertices[i].pos=vertices[i].pos-glm::vec3(mean);

        }
    }


    indexCount=0;
    for(unsigned int i=0;i<pclMesh.polygons.size();i++){
        if(pclMesh.polygons[i].vertices.size()==3){
            indexCount+=3;
        }
    }
    indices=new uint32_t[indexCount];

    for(unsigned int i=0;i<pclMesh.polygons.size();i++){
        if(pclMesh.polygons[i].vertices.size()==3){
            const std::vector<uint32_t> &vs = pclMesh.polygons[i].vertices;
            indices[i*3+0]=vs[0];
            indices[i*3+1]=vs[1];
            indices[i*3+2]=vs[2];
        }
    }


    offset=Eigen::Vector3f(-mean.x,-mean.y,-mean.z);

    scale=1.0f/(float)maxDistToCenter;

    //scale it:
    if(shiftToCenterAndNormalizeScale){
        for(uint32_t i=0;i<vertexCount;i++){
            vertices[i].pos=vertices[i].pos*scale;
        }
    }
}

DepthmapRendererModel::DepthmapRendererModel(const DepthmapRendererModel &obj)
{
    this->color=obj.color;
    this->geometry=obj.geometry;
    this->offset=obj.offset;
    this->scale=obj.scale;


    this->indexCount=obj.indexCount;
    this->indices=new uint32_t[this->indexCount];
    memcpy(this->indices,obj.indices,this->indexCount*sizeof(uint32_t));

    this->vertexCount=obj.vertexCount;
    this->vertices=new Vertex[this->vertexCount];
    memcpy(this->vertices,obj.vertices,this->vertexCount*sizeof(Vertex));

}

DepthmapRendererModel::~DepthmapRendererModel()
{
    delete[] vertices;
    delete[] indices;
}


///This was simon testing out the copy and swap idiom
void swap(DepthmapRendererModel& first, DepthmapRendererModel& second){
    using std::swap;
    swap(first.color,second.color);
    swap(first.geometry,second.geometry);
    swap(first.offset,second.offset);
    swap(first.scale,second.scale);

    swap(first.indexCount,second.indexCount);
    swap(first.indices,second.indices);

    swap(first.vertexCount,second.vertexCount);
    swap(first.vertices,second.vertices);
}

DepthmapRendererModel &DepthmapRendererModel::operator =(DepthmapRendererModel obj){
    swap(*this,obj);
    return *this;
}

void DepthmapRendererModel::loadToGPU(GLuint &VBO,GLuint &IBO)
{

    if(VBO){
        glDeleteBuffers(1,&VBO);
        glDeleteBuffers(1,&IBO);
    }

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex)*vertexCount, vertices, GL_STATIC_DRAW);


    glGenBuffers(1, &IBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*indexCount, indices, GL_STATIC_DRAW);


    //Make sure the shader is bound beforehand: (set the attribute location)


}


unsigned int DepthmapRendererModel::getIndexCount()
{
    return indexCount;
}

float DepthmapRendererModel::getScale(){
    return scale;
}

bool DepthmapRendererModel::hasColor()
{
    return color;
}

Eigen::Vector3f DepthmapRendererModel::getOffset(){
    return offset;
}

}
