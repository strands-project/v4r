
#include "output/roi.h"


namespace object_modeller
{
namespace output
{



void Roi::handleMouseMove(pcl::visualization::Camera camera, int screen_x, int screen_y)
{
    if (selectedPoint != NULL)
    {
        Eigen::Matrix4d view;
        Eigen::Matrix4d proj;
        camera.computeProjectionMatrix(proj);
        camera.computeViewMatrix(view);

        Eigen::Matrix4d inv_view = view.inverse();
        Eigen::Matrix4d inv_proj = proj.inverse();
        Eigen::Vector4d screen(
                    (2.0 * screen_x) / camera.window_size[0] - 1.0,
                    (2.0 * screen_y) / camera.window_size[1] - 1.0,
                    1, 1);

        Eigen::Vector4d far_point = inv_view * (inv_proj * screen);
        far_point /= far_point[3];

        screen[2] = 0;
        Eigen::Vector4d near_point = inv_view * (inv_proj * screen);
        near_point /= near_point[3];

        Eigen::Vector3f start(near_point[0], near_point[1], near_point[2]);
        Eigen::Vector3f end(far_point[0], far_point[1], far_point[2]);

        if (selectedPoint == translationPoint) handleTranslation(camera, start, end);

        if (selectedPoint == scalePointX) handleScale(camera, start, end, scalePointX);
        if (selectedPoint == scalePointY) handleScale(camera, start, end, scalePointY);
        if (selectedPoint == scalePointZ) handleScale(camera, start, end, scalePointZ);

        if (selectedPoint == rotPointX) handleRotation(camera, start, end, rotPointX);
        if (selectedPoint == rotPointY) handleRotation(camera, start, end, rotPointY);
        if (selectedPoint == rotPointZ) handleRotation(camera, start, end, rotPointZ);
    }
}

void Roi::handleRotation(pcl::visualization::Camera camera, Eigen::Vector3f ray_start, Eigen::Vector3f ray_end, pcl::PointXYZRGB *rotPoint)
{
    /*
    Eigen::Vector3f cam_up;//(camera.view[0], camera.view[1], camera.view[2]);
    Eigen::Vector3f rot_dir = originalPoint - originalTranslation;
    rot_dir.normalize();
    cam_up.normalize();

    Eigen::Vector3f rot_normal = cam_up.cross(rot_dir);
    rot_normal.normalize();
    */

    Eigen::Vector3f rot_normal = Eigen::Vector3f::Zero();

    if (rotPoint == rotPointX) rot_normal = Eigen::Vector3f(rotPointZ->x, rotPointZ->y, rotPointZ->z);
    if (rotPoint == rotPointY) rot_normal = Eigen::Vector3f(rotPointX->x, rotPointX->y, rotPointX->z);
    if (rotPoint == rotPointZ) rot_normal = Eigen::Vector3f(rotPointY->x, rotPointY->y, rotPointY->z);

    rot_normal -= originalTranslation;
    rot_normal.normalize();


    // intersect ray with plane lying in scale axis normal to camera up vector
    Eigen::ParametrizedLine<float, 3> line(ray_start, ray_start - ray_end);
    Eigen::Hyperplane<float, 3> plane(rot_normal, originalPoint);

    Eigen::Vector3f plane_target;
    float d = line.intersection(plane);
    plane_target = ray_start + (ray_start - ray_end) * d; // target is within plane but not on scale axis

    Eigen::Vector3f target_vec = originalTranslation - plane_target;
    target_vec.normalize();

    Eigen::Vector3f oldVec = originalTranslation - originalPoint;
    oldVec.normalize();

    Eigen::Quaternionf q;
    q.setFromTwoVectors(oldVec, target_vec);
    q.normalize();

    *rotation = q.toRotationMatrix() * originalRotation;
    rotation->normalize();

    /*
    Eigen::Vector3f target_point = originalTranslation - target_vec * 0.2;

    rotPoint->x = target_point[0];
    rotPoint->y = target_point[1];
    rotPoint->z = target_point[2];
    */

    updateRotationPoint(rotPointX);
    updateRotationPoint(rotPointY);
    updateRotationPoint(rotPointZ);

    updateScalePoint(scalePointX);
    updateScalePoint(scalePointY);
    updateScalePoint(scalePointZ);

    updateCube();
}

void Roi::handleScale(pcl::visualization::Camera camera, Eigen::Vector3f ray_start, Eigen::Vector3f ray_end, pcl::PointXYZRGB *scalePoint)
{
    Eigen::Vector3f cam_up(camera.view[0], camera.view[1], camera.view[2]);
    Eigen::Vector3f scale_dir = originalPoint - originalTranslation;
    Eigen::Vector3f scale_normal = cam_up.cross(scale_dir);
    scale_normal.normalize();
    scale_dir.normalize();


    // intersect ray with plane lying in scale axis normal to camera up vector
    Eigen::ParametrizedLine<float, 3> line(ray_start, ray_start - ray_end);
    Eigen::Hyperplane<float, 3> plane(scale_normal, originalPoint);

    Eigen::Vector3f plane_target;
    float d = line.intersection(plane);
    plane_target = ray_start + (ray_start - ray_end) * d; // target is within plane but not on scale axis


    Eigen::Vector3f target;
    //Eigen::Hyperplane<float, 3> plane2(cam_up, originalPoint);
    //target = plane2.projection(plane_target);
    Eigen::Hyperplane<float, 3> plane2(scale_dir, plane_target);
    Eigen::ParametrizedLine<float, 3> line2(originalPoint, scale_dir);
    d = line2.intersection(plane2);
    target = originalPoint + scale_dir * d;

    translationPoint->x += (target[0] - scalePoint->x) / 2.0;
    translationPoint->y += (target[1] - scalePoint->y) / 2.0;
    translationPoint->z += (target[2] - scalePoint->z) / 2.0;

    if (scalePoint != scalePointX) updateScalePoint(scalePointX);
    if (scalePoint != scalePointY) updateScalePoint(scalePointY);
    if (scalePoint != scalePointZ) updateScalePoint(scalePointZ);

    scalePoint->x = target[0];
    scalePoint->y = target[1];
    scalePoint->z = target[2];

    updateRotationPoint(rotPointX);
    updateRotationPoint(rotPointY);
    updateRotationPoint(rotPointZ);

    updateCube();
}

void Roi::updateRotationPoint(pcl::PointXYZRGB *rotPoint)
{
    rotPoint->x = translationPoint->x;
    rotPoint->y = translationPoint->y;
    rotPoint->z = translationPoint->z;

    Eigen::Vector3f dir;

    if (rotPoint == rotPointX) dir = Eigen::Vector3f(0.2, 0.0, 0.0);
    if (rotPoint == rotPointY) dir = Eigen::Vector3f(0.0, 0.2, 0.0);
    if (rotPoint == rotPointZ) dir = Eigen::Vector3f(0.0, 0.0, 0.2);

    dir.normalize();

    /*
    Eigen::Quaternionf q;
    q.setFromTwoVectors(Eigen::Vector3f(0.0, 1.0, 0.0), rotation);
    q.normalize();
    */

    dir = rotation->toRotationMatrix() * dir;

    rotPoint->x += dir[0] * 0.2;
    rotPoint->y += dir[1] * 0.2;
    rotPoint->z += dir[2] * 0.2;
}

void Roi::updateScalePoint(pcl::PointXYZRGB *scalePoint)
{
    Eigen::Vector3f result = Eigen::Vector3f::Zero();

    if (scalePoint == scalePointX) result[0] -= dimension->x() / 2.0;
    if (scalePoint == scalePointY) result[1] -= dimension->y() / 2.0;
    if (scalePoint == scalePointZ) result[2] -= dimension->z() / 2.0;

    result = rotation->toRotationMatrix() * result;

    scalePoint->x = result[0];
    scalePoint->y = result[1];
    scalePoint->z = result[2];

    scalePoint->x += translationPoint->x;
    scalePoint->y += translationPoint->y;
    scalePoint->z += translationPoint->z;
}

void Roi::handleTranslation(pcl::visualization::Camera camera, Eigen::Vector3f ray_start, Eigen::Vector3f ray_end)
{
    // intersection
    Eigen::Vector3f target;
    Eigen::Vector3f cam_dir(camera.focal[0] - camera.pos[0], camera.focal[1] - camera.pos[1], camera.focal[2] - camera.pos[2]);
    cam_dir.normalize();

    Eigen::ParametrizedLine<float, 3> line(ray_start, ray_start - ray_end);
    Eigen::Hyperplane<float, 3> plane(cam_dir, originalPoint);

    float d = line.intersection(plane);
    target = ray_start + (ray_start - ray_end) * d;

    translationPoint->x = target[0];
    translationPoint->y = target[1];
    translationPoint->z = target[2];

    updateScalePoint(scalePointX);
    updateScalePoint(scalePointY);
    updateScalePoint(scalePointZ);

    updateRotationPoint(rotPointX);
    updateRotationPoint(rotPointY);
    updateRotationPoint(rotPointZ);

    updateCube();
}

void Roi::updateCube()
{
    translation->x() = translationPoint->x;
    translation->y() = translationPoint->y;
    translation->z() = translationPoint->z;

    Eigen::Vector3f dimX(translationPoint->x - scalePointX->x, translationPoint->y - scalePointX->y, translationPoint->z - scalePointX->z);
    dimension->x() = dimX.norm() * 2.0;

    Eigen::Vector3f dimY(translationPoint->x - scalePointY->x, translationPoint->y - scalePointY->y, translationPoint->z - scalePointY->z);
    dimension->y() = dimY.norm() * 2.0;

    Eigen::Vector3f dimZ(translationPoint->x - scalePointZ->x, translationPoint->y - scalePointZ->y, translationPoint->z - scalePointZ->z);
    dimension->z() = dimZ.norm() * 2.0;
}

void Roi::selectPoint(pcl::visualization::Camera camera, int screen_x, int screen_y)
{
    Eigen::Matrix4d view;
    Eigen::Matrix4d proj;
    camera.computeProjectionMatrix(proj);
    camera.computeViewMatrix(view);

    selectedPoint = NULL;
    float bestDistance = 5.0f;

    for (int i=0;i<cloud->size();i++)
    {
        pcl::PointXYZRGB *cp = &(cloud->at(i));

        Eigen::Vector4d p(cp->x, cp->y, cp->z, 1.0);
        Eigen::Vector4d p_proj = proj * (view * p);

        p_proj /= p_proj[3];

        p_proj[0] = (p_proj[0] + 1.0) * camera.window_size[0] / 2.0;
        p_proj[1] = (p_proj[1] + 1.0) * camera.window_size[1] / 2.0;

        float dist = Eigen::Vector2f(screen_x - p_proj[0], screen_y - p_proj[1]).norm();

        if (dist < bestDistance)
        {
            selectedPoint = cp;
            originalPoint = Eigen::Vector3f(cp->x, cp->y, cp->z);
            originalTranslation = *translation;
            originalRotation = *rotation;
            bestDistance = dist;
        }
    }
}

Roi::Roi(Eigen::Vector3f *dimension, Eigen::Vector3f *translation, Eigen::Quaternionf *rotation)
{
    this->dimension = dimension;
    this->rotation = rotation;
    this->translation = translation;

    //updateCube();

    selectedPoint = NULL;
    highlightPoint = NULL;

    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

    for (int i=0;i<7;i++)
    {
        pcl::PointXYZRGB p(100, 100, 100);
        cloud->push_back(p);
    }

    translationPoint = &(cloud->at(0));

    scalePointX = &(cloud->at(1));
    scalePointY = &(cloud->at(2));
    scalePointZ = &(cloud->at(3));

    rotPointX = &(cloud->at(4));
    rotPointY = &(cloud->at(5));
    rotPointZ = &(cloud->at(6));

    translationPoint->x = translation->x();
    translationPoint->y = translation->y();
    translationPoint->z = translation->z();
    translationPoint->r = 200;
    translationPoint->g = 200;
    translationPoint->b = 200;

    rotPointX->x = translation->x() + 0.2;
    rotPointX->y = translation->y();
    rotPointX->z = translation->z();
    rotPointX->r = 200;

    rotPointY->x = translation->x();
    rotPointY->y = translation->y() + 0.2;
    rotPointY->z = translation->z();
    rotPointY->g = 200;

    rotPointZ->x = translation->x();
    rotPointZ->y = translation->y();
    rotPointZ->z = translation->z() + 0.2;
    rotPointZ->b = 200;

    scalePointX->r = 200;
    scalePointY->g = 200;
    scalePointZ->b = 200;

    updateScalePoint(scalePointX);
    updateScalePoint(scalePointY);
    updateScalePoint(scalePointZ);

    updateRotationPoint(rotPointX);
    updateRotationPoint(rotPointY);
    updateRotationPoint(rotPointZ);
}

}
}
