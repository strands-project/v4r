#include <pcl/common/transforms.h>
#include "ceres/cost_function.h"
#include "ceres/ceres.h"
#include <ceres/rotation.h>
#include "ceres/conditioned_cost_function.h"
#include <boost/scoped_ptr.hpp>
#include <v4r/common/miscellaneous.h>
#include <v4r/registration/MvLMIcp.h>

//#define USE_QUATERNIONS_AUTO_DIFF

template<typename T>
struct JetOps {
    static bool IsScalar() {
        return true;
    }
    static T GetScalar(const T& t) {
        return t;
    }
    static void SetScalar(const T& scalar, T* t) {
        *t = scalar;
    }
    static void ScaleDerivative(double scale_by, T *value) {
        // For double, there is no derivative to scale.
        (void) scale_by; // Ignored.
        (void) value; // Ignored.
    }
};
template<typename T, int N>
struct JetOps<ceres::Jet<T, N> > {
    static bool IsScalar() {
        return false;
    }
    static T GetScalar(const ceres::Jet<T, N>& t) {
        return t.a;
    }
    static void SetScalar(const T& scalar, ceres::Jet<T, N>* t) {
        t->a = scalar;
    }
    static void ScaleDerivative(double scale_by, ceres::Jet<T, N> *value) {
        value->v *= scale_by;
    }
};
template<typename FunctionType, int kNumArgs, typename ArgumentType>
struct Chain {
    static ArgumentType Rule(const FunctionType &f,
                             const FunctionType dfdx[kNumArgs],
                             const ArgumentType x[kNumArgs]) {
        // In the default case of scalars, there's nothing to do since there are no
        // derivatives to propagate.
        (void) dfdx; // Ignored.
        (void) x; // Ignored.
        return f;
    }
};
// XXX Add documentation here!
template<typename FunctionType, int kNumArgs, typename T, int N>
struct Chain<FunctionType, kNumArgs, ceres::Jet<T, N> > {
    static ceres::Jet<T, N> Rule(const FunctionType &f,
                                 const FunctionType dfdx[kNumArgs],
                                 const ceres::Jet<T, N> x[kNumArgs])
    {
        // x is itself a function of another variable ("z"); what this function
        // needs to return is "f", but with the derivative with respect to z
        // attached to the jet. So combine the derivative part of x's jets to form
        // a Jacobian matrix between x and z (i.e. dx/dz).
        Eigen::Matrix<T, kNumArgs, N> dxdz;
        for (int i = 0; i < kNumArgs; ++i)
        {
            dxdz.row(i) = x[i].v.transpose();
        }

        // Map the input gradient dfdx into an Eigen row vector.
        Eigen::Map<const Eigen::Matrix<FunctionType, 1, kNumArgs> > vector_dfdx(dfdx, 1, kNumArgs);

        // Now apply the chain rule to obtain df/dz. Combine the derivative with
        // the scalar part to obtain f with full derivative information.
        ceres::Jet<T, N> jet_f;
        jet_f.a = f;
        jet_f.v = vector_dfdx.template cast<T>() * dxdz; // Also known as dfdz.
        return jet_f;
    }
};

inline Eigen::Matrix3d coolquat_to_mat(Eigen::Quaterniond & q)
{
    Eigen::Matrix3d R;
    double x2 = q.x() * q.x();
    double y2 = q.y() * q.y();
    double z2 = q.z() * q.z();
    double r2 = q.w() * q.w();

    R(0,0) = r2 + x2 - y2 - z2;
    R(1,1) = r2 - x2 + y2 - z2;
    R(2,2) = r2 - x2 - y2 + z2;

    double xy = q.x()  * q.y();
    double yz = q.y() *  q.z();
    double zx =  q.z() * q.x();
    double rx = q.w() * q.x();
    double ry = q.w() * q.y();
    double rz = q.w() *  q.z();

    R(0,1) = 2 * (xy + rz);
    R(0,2) = 2 * (zx - ry);
    R(1,2) = 2 * (yz + rx);
    R(1,0) = 2 * (xy - rz);
    R(2,0) = 2 * (zx + ry);
    R(2,1) = 2 * (yz - rx);
    return R;
}

///numeric diff cost function

template<class PointT>
class RegistrationCostFunctor
{

private:
    typename v4r::Registration::MvLMIcp<PointT> * nl_icp;

    //view indices considered for this block
    int h_;
    int k_;

public:
    RegistrationCostFunctor(typename v4r::Registration::MvLMIcp<PointT> * data, int h, int k)
    {
        nl_icp = data;
        h_ = h;
        k_ = k;
    }

    bool operator()(const double* const ph,
                    const double* const pk,
                    double* residuals) const
    {
        Eigen::Vector3d trans_h = Eigen::Vector3d(ph[3], ph[4], ph[5]);
        Eigen::Vector3d trans_k = Eigen::Vector3d(pk[3], pk[4], pk[5]);

        Eigen::Matrix3d R_h, R_k;
        Eigen::Matrix4f T_h, T_k;
        T_h.setIdentity();
        T_k.setIdentity();

        ceres::AngleAxisToRotationMatrix<double>(ph, R_h.data());
        ceres::AngleAxisToRotationMatrix<double>(pk, R_k.data());

        Eigen::Matrix4f T;
        T_h.block<3,3>(0,0) = R_h.cast<float>();
        T_h.block<3,1>(0,3) = trans_h.cast<float>();
        T_k.block<3,3>(0,0) = R_k.cast<float>();
        T_k.block<3,1>(0,3) = trans_k.cast<float>();
        T = T_k.inverse() * T_h;

        for(size_t i=0; i < nl_icp->clouds_transformed_with_ip_[h_]->points.size(); i++)
        {
            //transform point into CS of k_ and find distance to closest point in k_
            PointT p;
            p.getVector4fMap() = T * nl_icp->clouds_transformed_with_ip_[h_]->points[i].getVector4fMap();

            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;
            if (nl_icp->octrees_[k_]->nearestKSearch (p, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                residuals[i] = std::min(sqrt(pointNKNSquaredDistance[0]), nl_icp->max_correspondence_distance_);
            }
            else
            {
                residuals[i] = 0.0;
            }
        }

        return true;
    }
};

template<class PointT>
class RegistrationCostFunctionAutoDiff
{

private:
    typename v4r::Registration::MvLMIcp<PointT> * nl_icp;

    //view indices considered for this block
    int h_;
    int k_;

public:

    RegistrationCostFunctionAutoDiff(typename v4r::Registration::MvLMIcp<PointT> * data, int h, int k)
    {
        nl_icp = data;
        h_ = h;
        k_ = k;
    }

    template <typename T>
    bool operator()(const T* const params_h,
                    const T* const params_k,
                    T* residual) const
    {

        //params_h and params_k (first 3 elements are rotation, 4 to 6 are translation)
        Eigen::Matrix<T, 4, 4> T_h, T_k, T_hk;
        T_h.setIdentity();
        T_k.setIdentity();

        Eigen::Matrix<T, 3, 3> R_h, R_k, R_hk;

        ceres::AngleAxisToRotationMatrix(params_h, R_h.data());
        ceres::AngleAxisToRotationMatrix(params_k, R_k.data());

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                T_h(i,j) = R_h(i,j);
                T_k(i,j) = R_k(i,j);
            }
        }

        for(int k=0; k < 3; k++)
        {
            T_h(k, 3) = *(params_h + 3 + k);
            T_k(k, 3) = *(params_k + 3 + k);
        }

        T_hk = T_k.inverse() * T_h;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                R_hk(i,j) = T_hk(i,j);
                R_hk(i,j) = T_hk(i,j);
            }
        }

        for(size_t i=0; i < nl_icp->clouds_transformed_with_ip_[h_]->points.size(); i++)
        {
            Eigen::Vector4f p = nl_icp->clouds_transformed_with_ip_[h_]->points[i].getVector4fMap();

            T w = T(1);

            //transform p to original CS of k, then we can add the gradient from k at that location
            Eigen::Matrix<T, 4, 1> pt;
            pt[0] = T(p[0]); pt[1] = T(p[1]); pt[2] = T(p[2]); pt[3] = T(1);
            pt = T_hk * pt;

            //residual is simply the distance from pt to the NN in point cloud k_
            float x,y,z;
            x = float(JetOps<T>::GetScalar(pt[0]));
            y = float(JetOps<T>::GetScalar(pt[1]));
            z = float(JetOps<T>::GetScalar(pt[2]));

            PointT p_at_k;
            p_at_k.getVector3fMap() = Eigen::Vector3f(x,y,z);

            float dist;
            int idx;
            nl_icp->distance_transforms_[k_]->getCorrespondence(p_at_k, &idx, &dist, 0, 0);

            //float dist = nl_icp->distance_transforms_[k_]->getCorrespondence(x,y,z);

            if(dist < 0 || dist > nl_icp->max_correspondence_distance_)
            {
                residual[i] = T(nl_icp->max_correspondence_distance_);
                continue;
            }

            if(nl_icp->normals_.size() == nl_icp->clouds_.size())
            {
                Eigen::Vector3f n_h, n_k;
                n_h = nl_icp->normals_transformed_with_ip_[h_]->points[i].getNormalVector3fMap();
                n_k = nl_icp->normals_transformed_with_ip_[k_]->points[idx].getNormalVector3fMap();

                Eigen::Matrix<T, 3, 1> n_hT, n_kT;
                n_hT[0] = T(n_h[0]); n_hT[1] = T(n_h[1]); n_hT[2] = T(n_h[2]);
                n_kT[0] = T(n_k[0]); n_kT[1] = T(n_k[1]); n_kT[2] = T(n_k[2]);

                n_hT = R_hk * n_hT;


                if(n_hT.dot(n_kT) < T(nl_icp->normal_dot_))
                {
                    w = T(0);
                }
            }

            //use idx to compute the distance
            /*Eigen::Vector3f nn_k = nl_icp->clouds_transformed_with_ip_[k_]->points[idx].getVector3fMap();
            Eigen::Matrix<T, 4, 1> nn_ptk;
            nn_ptk[0] = T(nn_k[0]); nn_ptk[1] = T(nn_k[1]); nn_ptk[2] = T(nn_k[2]); nn_ptk[3] = T(1);*/

            //residual[i] = T( (nn_ptk - pt).norm() );
            residual[i] = T(dist) * w;

            //propagate gradient if ceres asking for derivatives
            if (!JetOps<T>::IsScalar())
            {
                //For the derivative case, sample the gradient as well.
                float sample[4];
                sample[0] = dist;

                //TODO: trilinear interpolation
                Eigen::Vector3f dxyz;
                //nl_icp->distance_transforms_[k_]->getDerivatives(p_at_k, dxyz);
                nl_icp->distance_transforms_[k_]->getDerivatives(x, y, z, dxyz);

                if(!pcl_isfinite(dxyz[0]))
                {
                    residual[i] = T(0);
                    continue;
                }

                for(int k=0; k < 3; k++)
                    sample[1+k] = dxyz[k];

                T xyz[3] = { pt[0], pt[1], pt[2] };

                residual[i] = Chain<float, 3, T>::Rule(sample[0], sample + 1, xyz);
            }

        }

        return true;
    }
};

/// cost function (residuals + jacobian)
/// acts on a pair of views

template<class PointT>
class RegistrationCostFunction
        : public ceres::CostFunction {

private:
    typename v4r::Registration::MvLMIcp<PointT> * nl_icp;

    //view indices considered for this block
    int h_;
    int k_;
    #ifdef CERES_VERSION_LESS_1_9_0
    std::vector<short> * param_blocks_sizes_;
    #else
    std::vector<int> * param_blocks_sizes_;
    #endif

public:

    RegistrationCostFunction(typename v4r::Registration::MvLMIcp<PointT> * data, int h, int k)
    {
        nl_icp = data;
        h_ = h;
        k_ = k;

        set_num_residuals( static_cast<int>(data->clouds_[h]->points.size()));
        param_blocks_sizes_ = mutable_parameter_block_sizes();
        param_blocks_sizes_->resize(2, 7);
    }

    virtual ~RegistrationCostFunction() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const
    {


        Eigen::Quaterniond rot_h(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond rot_k(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Vector3d trans_h = Eigen::Vector3d(parameters[0][4], parameters[0][5], parameters[0][6]);
        Eigen::Vector3d trans_k = Eigen::Vector3d(parameters[1][4], parameters[1][5], parameters[1][6]);

        Eigen::Matrix3d R_h, R_k;
        Eigen::Matrix4f T_h, T_k;
        T_h.setIdentity();
        T_k.setIdentity();

        R_h = coolquat_to_mat(rot_h);
        R_h /= rot_h.squaredNorm();
        R_k = coolquat_to_mat(rot_k);
        R_k /= rot_k.squaredNorm();

        Eigen::Matrix4f T;
        T_h.block<3,3>(0,0) = R_h.cast<float>();
        T_h.block<3,1>(0,3) = trans_h.cast<float>();
        T_k.block<3,3>(0,0) = R_k.cast<float>();
        T_k.block<3,1>(0,3) = trans_k.cast<float>();
        T = T_k.inverse() * T_h;

        //        std::cout << h_ << " " << k_ << std::endl;
        //std::cout << T << std::endl;

        //        for(int k=0; k < (*param_blocks_sizes_)[0]; k++)
        //        {
        //            std::cout << parameters[0][k] << " ";
        //        }

        //        std::cout << std::endl;

        //        for(int k=0; k < (*param_blocks_sizes_)[1]; k++)
        //        {
        //            std::cout << parameters[1][k] << " ";
        //        }

        //        std::cout << std::endl;

        float step = 0.001f;
        //double huber_sigma = 0.003f * 1.345;

        //compute residuals and jacobians
        assert(num_residuals() == (int)nl_icp->clouds_transformed_with_ip_[h_]->points.size());

        //ceres::CauchyLoss cauchy(nl_icp->max_correspondence_distance_ / 2.0);

        //GPU speed up
        // this parts can be speed up on the GPU by doing the nearest neighbour search of the transformed point
        // as well as finite differences on the GPU (7NN for each point in h^th cloud)
        // todo: test improvement using PCL octrees implementation and then decide if worthy implementing or not

        for(size_t i=0; i < nl_icp->clouds_transformed_with_ip_[h_]->points.size(); i++)
        {
            //transform point into CS of k_ and find distance to closest point in k_
            PointT p;
            p.getVector4fMap() = T * nl_icp->clouds_transformed_with_ip_[h_]->points[i].getVector4fMap();

            //std::cout << nl_icp->clouds_transformed_with_ip_[h_]->points[i].getVector4fMap() << std::endl;

            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;
            if (nl_icp->octrees_[k_]->nearestKSearch (p, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                residuals[i] = std::min(sqrt(pointNKNSquaredDistance[0]), nl_icp->max_correspondence_distance_);
                //residuals[i] = sqrt(pointNKNSquaredDistance[0]);
            }
            else
            {
                residuals[i] = 0.0;
            }

            float w = 1.f;
            if(nl_icp->weights_.size() == nl_icp->clouds_.size())
            {
                w = nl_icp->weights_[h_][i] * nl_icp->weights_[k_][pointIdxNKNSearch[0]];
            }

            if(nl_icp->normals_.size() == nl_icp->clouds_.size())
            {
                Eigen::Vector3f n_h, n_k;
                n_h = T.block<3,3>(0,0) * nl_icp->normals_transformed_with_ip_[h_]->points[i].getNormalVector3fMap();
                n_k = nl_icp->normals_transformed_with_ip_[k_]->points[pointIdxNKNSearch[0]].getNormalVector3fMap();

                if(n_h.dot(n_k) < nl_icp->normal_dot_)
                {
                    w = 0;
                }
            }

            residuals[i] *= w;

            if(jacobians != NULL)
            {
                //jacobian
                float x,y,z;
                float Lx,Ly,Lz;
                float qxh,qyh,qzh,qrh,txh,tyh,tzh;
                float qxk,qyk,qzk,qrk,txk,tyk,tzk;

                qxh = parameters[0][0];
                qyh = parameters[0][1];
                qzh = parameters[0][2];
                qrh = parameters[0][3];
                txh = parameters[0][4];
                tyh = parameters[0][5];
                tzh = parameters[0][6];

                qxk = parameters[1][0];
                qyk = parameters[1][1];
                qzk = parameters[1][2];
                qrk = parameters[1][3];
                txk = parameters[1][4];
                tyk = parameters[1][5];
                tzk = parameters[1][6];

                PointT p_plus, p_minus;
                float distance_plus, distance_minus;
                distance_plus = distance_minus = 0.f;

                { //X
                    p_plus = p;
                    p_minus = p;
                    p_plus.x += step / 2.f;
                    p_minus.x -= step / 2.f;

                    if (nl_icp->octrees_[k_]->nearestKSearch (p_plus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        distance_plus = std::min(sqrt(pointNKNSquaredDistance[0]), nl_icp->max_correspondence_distance_);
                    }

                    if (nl_icp->octrees_[k_]->nearestKSearch (p_minus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        distance_minus = std::min(sqrt(pointNKNSquaredDistance[0]), nl_icp->max_correspondence_distance_);
                    }

                    Lx = distance_plus - distance_minus;
                    /*double dplus;
                    double dminus;
                    cauchy.Evaluate(distance_plus, &dplus);
                    cauchy.Evaluate(distance_minus, &dminus);
                    Lx = dplus - dminus;*/


                }

                { //Y
                    p_plus = p;
                    p_minus = p;
                    p_plus.y += step / 2.f;
                    p_minus.y -= step / 2.f;

                    if (nl_icp->octrees_[k_]->nearestKSearch (p_plus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        //idx_match_plus = pointIdxNKNSearch[0];
                        distance_plus = std::min(sqrt(pointNKNSquaredDistance[0]), nl_icp->max_correspondence_distance_);
                    }

                    if (nl_icp->octrees_[k_]->nearestKSearch (p_minus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        //idx_match_minus = pointIdxNKNSearch[0];
                        distance_minus = std::min(sqrt(pointNKNSquaredDistance[0]), nl_icp->max_correspondence_distance_);
                    }

                    Ly = distance_plus - distance_minus;

                    /*double dplus;
                    double dminus;
                    cauchy.Evaluate(distance_plus, &dplus);
                    cauchy.Evaluate(distance_minus, &dminus);
                    Ly = dplus - dminus;*/
                }

                { //Z
                    p_plus = p;
                    p_minus = p;
                    p_plus.z += step / 2.f;
                    p_minus.z -= step / 2.f;

                    if (nl_icp->octrees_[k_]->nearestKSearch (p_plus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        distance_plus = std::min(sqrt(pointNKNSquaredDistance[0]), nl_icp->max_correspondence_distance_);
                    }

                    if (nl_icp->octrees_[k_]->nearestKSearch (p_minus, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        distance_minus = std::min(sqrt(pointNKNSquaredDistance[0]), nl_icp->max_correspondence_distance_);
                    }

                    Lz = distance_plus - distance_minus;

                    /*double dplus;
                    double dminus;
                    cauchy.Evaluate(distance_plus, &dplus);
                    cauchy.Evaluate(distance_minus, &dminus);
                    Lz = dplus - dminus;*/

                }

                x = p.x;
                y = p.y;
                z = p.z;

                //fill jacobians for parameter blocks
                if(jacobians[0] != NULL)
                {
                    float j1,j2,j3,j4,j5,j6,j7;
                    float t2 = qrk*qrk;
                    float t3 = qxk*qxk;
                    float t4 = qyk*qyk;
                    float t5 = qzk*qzk;
                    float t6 = qrh*qrh;
                    float t7 = qxh*qxh;
                    float t8 = qyh*qyh;
                    float t9 = qzh*qzh;
                    float t10 = t6+t7+t8+t9;
                    float t11 = 1.0/t10;
                    float t12 = t2+t3+t4+t5;
                    float t13 = 1.0/t12;
                    float t14 = t2+t3-t4-t5;
                    float t15 = qrk*qyk*2.0;
                    float t16 = qxk*qzk*2.0;
                    float t17 = t15+t16;
                    float t18 = 1.0/(t10*t10);
                    float t19 = qrk*qzk*2.0;
                    float t21 = qxk*qyk*2.0;
                    float t20 = t19-t21;
                    float t22 = qrh*qzh*2.0;
                    float t23 = qrh*qyh*2.0;
                    float t24 = qxh*qzh*2.0;
                    float t25 = qrh*qxh*2.0;
                    float t26 = qxh*qyh*2.0;
                    float t27 = t22-t26;
                    float t28 = t2-t3+t4-t5;
                    float t29 = t19+t21;
                    float t30 = t6+t7-t8-t9;
                    float t31 = t23+t24;
                    float t32 = qrk*qxk*2.0;
                    float t34 = qyk*qzk*2.0;
                    float t33 = t32-t34;
                    float t35 = t6-t7+t8-t9;
                    float t36 = qyh*qzh*2.0;
                    float t37 = t22+t26;
                    float t38 = t25+t36;
                    float t39 = t6-t7-t8+t9;
                    float t40 = t23-t24;
                    float t41 = t2-t3-t4+t5;
                    float t42 = t15-t16;
                    float t43 = t32+t34;
                    float t44 = t25-t36;
                    float t45 = qyh*t11*t13*t14*2.0;
                    float t46 = qxh*t11*t13*t20*2.0;
                    float t47 = qyh*t11*t13*t20*2.0;
                    float t48 = qxh*t11*t13*t28*2.0;
                    float t49 = qyh*t11*t13*t28*2.0;
                    float t50 = qxh*t11*t13*t29*2.0;
                    float t51 = qrh*t11*t13*t41*2.0;
                    float t52 = qxh*t11*t13*t43*2.0;
                    float t53 = qyh*t11*t13*t42*2.0;
                    float t54 = qzh*t11*t13*t41*2.0;
                    float t55 = qyh*t11*t13*t43*2.0;
                    float t56 = qrh*t11*t13*t20*2.0;
                    float t57 = qxh*t11*t13*t17*2.0;
                    float t58 = qrh*t11*t13*t14*2.0;
                    float t59 = qyh*t11*t13*t17*2.0;
                    float t60 = qzh*t11*t13*t20*2.0;
                    float t61 = qxh*t11*t13*t14*2.0;
                    float t62 = qzh*t11*t13*t17*2.0;
                    float t63 = qrh*t11*t13*t28*2.0;
                    float t64 = qxh*t11*t13*t33*2.0;
                    float t65 = qzh*t11*t13*t29*2.0;
                    float t66 = qzh*t11*t13*t28*2.0;
                    float t67 = qyh*t11*t13*t33*2.0;
                    float t68 = qzh*t11*t13*t33*2.0;
                    float t69 = qxh*t11*t13*t41*2.0;
                    float t70 = qzh*t11*t13*t42*2.0;
                    float t71 = qyh*t11*t13*t41*2.0;
                    float t72 = qzh*t11*t13*t14*2.0;

                    j1 = Ly*(x*(t49+t50-qzh*t11*t13*t33*2.0+qxh*t13*t18*t27*t28*2.0-qxh*t13*t18*t29*t30*2.0+qxh*t13*t18*t31*t33*2.0)+z*(t63+t64+t65-qxh*t13*t18*t28*t38*2.0+qxh*t13*t18*t29*t40*2.0+qxh*t13*t18*t33*t39*2.0)-y*(t48-qrh*t11*t13*t33*2.0-qyh*t11*t13*t29*2.0+qxh*t13*t18*t28*t35*2.0+qxh*t13*t18*t29*t37*2.0+qxh*t13*t18*t33*t44*2.0))-Lz*(-x*(t54+t55-qxh*t11*t13*t42*2.0+qxh*t13*t18*t27*t43*2.0+qxh*t13*t18*t30*t42*2.0-qxh*t13*t18*t31*t41*2.0)+z*(t69+t70-qrh*t11*t13*t43*2.0+qxh*t13*t18*t39*t41*2.0+qxh*t13*t18*t38*t43*2.0+qxh*t13*t18*t40*t42*2.0)+y*(t51+t52+t53+qxh*t13*t18*t35*t43*2.0-qxh*t13*t18*t37*t42*2.0-qxh*t13*t18*t41*t44*2.0))-Lx*(z*(t56+t57-qzh*t11*t13*t14*2.0+qxh*t13*t17*t18*t39*2.0-qxh*t13*t14*t18*t40*2.0-qxh*t13*t18*t20*t38*2.0)-y*(t45+t46-qrh*t11*t13*t17*2.0+qxh*t13*t17*t18*(t25-qyh*qzh*2.0)*2.0-qxh*t13*t14*t18*t37*2.0+qxh*t13*t18*t20*t35*2.0)+x*(t47-qxh*t11*t13*t14*2.0-qzh*t11*t13*t17*2.0+qxh*t13*t14*t18*t30*2.0+qxh*t13*t18*t20*t27*2.0+qxh*t13*t17*t18*t31*2.0));
                    j2 = Ly*(z*(t66+t67-qrh*t11*t13*t29*2.0-qyh*t13*t18*t28*t38*2.0+qyh*t13*t18*t29*t40*2.0+qyh*t13*t18*t33*t39*2.0)-y*(-t49-t50+t68+qyh*t13*t18*t28*t35*2.0+qyh*t13*t18*t29*t37*2.0+qyh*t13*t18*t33*t44*2.0)+x*(t48-qrh*t11*t13*t33*2.0-qyh*t11*t13*t29*2.0+qyh*t13*t18*t27*t28*2.0-qyh*t13*t18*t29*t30*2.0+qyh*t13*t18*t31*t33*2.0))-Lx*(x*(t45+t46-qrh*t11*t13*t17*2.0+qyh*t13*t14*t18*t30*2.0+qyh*t13*t18*t20*t27*2.0+qyh*t13*t17*t18*t31*2.0)+z*(t58+t59+t60+qyh*t13*t17*t18*t39*2.0-qyh*t13*t14*t18*t40*2.0-qyh*t13*t18*t20*t38*2.0)-y*(-t47+t61+t62-qyh*t13*t14*t18*t37*2.0+qyh*t13*t18*t20*t35*2.0+qyh*t13*t17*t18*t44*2.0))+Lz*(y*(t54+t55-qxh*t11*t13*t42*2.0-qyh*t13*t18*t35*t43*2.0+qyh*t13*t18*t37*t42*2.0+qyh*t13*t18*t41*t44*2.0)+x*(t51+t52+t53+qyh*t13*t18*t27*t43*2.0+qyh*t13*t18*t30*t42*2.0-qyh*t13*t18*t31*t41*2.0)-z*(t71-qrh*t11*t13*t42*2.0-qzh*t11*t13*t43*2.0+qyh*t13*t18*t39*t41*2.0+qyh*t13*t18*t38*t43*2.0+qyh*t13*t18*t40*t42*2.0));
                    j3 = Lz*(x*(t69+t70-qrh*t11*t13*t43*2.0+qzh*t13*t18*t27*t43*2.0+qzh*t13*t18*t30*t42*2.0-qzh*t13*t18*t31*t41*2.0)+y*(t71-qrh*t11*t13*t42*2.0-qzh*t11*t13*t43*2.0-qzh*t13*t18*t35*t43*2.0+qzh*t13*t18*t37*t42*2.0+qzh*t13*t18*t41*t44*2.0)-z*(-t54-t55+qxh*t11*t13*t42*2.0+qzh*t13*t18*t39*t41*2.0+qzh*t13*t18*t38*t43*2.0+qzh*t13*t18*t40*t42*2.0))-Ly*(y*(t66+t67-qrh*t11*t13*t29*2.0+qzh*t13*t18*t28*t35*2.0+qzh*t13*t18*t29*t37*2.0+qzh*t13*t18*t33*t44*2.0)+x*(t63+t64+t65-qzh*t13*t18*t27*t28*2.0+qzh*t13*t18*t29*t30*2.0-qzh*t13*t18*t31*t33*2.0)-z*(t49+t50-t68-qzh*t13*t18*t28*t38*2.0+qzh*t13*t18*t29*t40*2.0+qzh*t13*t18*t33*t39*2.0))+Lx*(-x*(-t56-t57+t72+qzh*t13*t14*t18*t30*2.0+qzh*t13*t18*t20*t27*2.0+qzh*t13*t17*t18*t31*2.0)+y*(t58+t59+t60-qzh*t13*t14*t18*t37*2.0+qzh*t13*t18*t20*t35*2.0+qzh*t13*t17*t18*t44*2.0)+z*(-t47+t61+t62-qzh*t13*t17*t18*t39*2.0+qzh*t13*t14*t18*t40*2.0+qzh*t13*t18*t20*t38*2.0));
                    j4 = -Lx*(z*(t45+t46-qrh*t11*t13*t17*2.0+qrh*t13*t17*t18*t39*2.0-qrh*t13*t14*t18*t40*2.0-qrh*t13*t18*t20*t38*2.0)-x*(t58+t59+t60-qrh*t13*t14*t18*t30*2.0-qrh*t13*t18*t20*t27*2.0-qrh*t13*t17*t18*t31*2.0)+y*(t56+t57-t72+qrh*t13*t14*t18*t37*2.0-qrh*t13*t18*t20*t35*2.0-qrh*t13*t17*t18*t44*2.0))+Ly*(-x*(t66+t67-qrh*t11*t13*t29*2.0-qrh*t13*t18*t27*t28*2.0+qrh*t13*t18*t29*t30*2.0-qrh*t13*t18*t31*t33*2.0)+y*(t63+t64+t65-qrh*t13*t18*t28*t35*2.0-qrh*t13*t18*t29*t37*2.0-qrh*t13*t18*t33*t44*2.0)+z*(t48-qrh*t11*t13*t33*2.0-qyh*t11*t13*t29*2.0-qrh*t13*t18*t28*t38*2.0+qrh*t13*t18*t29*t40*2.0+qrh*t13*t18*t33*t39*2.0))+Lz*(-y*(t69+t70-qrh*t11*t13*t43*2.0+qrh*t13*t18*t35*t43*2.0-qrh*t13*t18*t37*t42*2.0-qrh*t13*t18*t41*t44*2.0)+z*(t51+t52+t53-qrh*t13*t18*t39*t41*2.0-qrh*t13*t18*t38*t43*2.0-qrh*t13*t18*t40*t42*2.0)+x*(t71-qrh*t11*t13*t42*2.0-qzh*t11*t13*t43*2.0+qrh*t13*t18*t27*t43*2.0+qrh*t13*t18*t30*t42*2.0-qrh*t13*t18*t31*t41*2.0));
                    j5 = Lx*t13*t14+Ly*t13*t29-Lz*t13*t42;
                    j6 = -Lx*t13*t20+Ly*t13*t28+Lz*t13*t43;
                    j7 = Lx*t13*t17-Ly*t13*t33+Lz*t13*t41;

                    jacobians[0][i * (*param_blocks_sizes_)[0]    ] = w * j1;
                    jacobians[0][i * (*param_blocks_sizes_)[0] + 1] = w * j2;
                    jacobians[0][i * (*param_blocks_sizes_)[0] + 2] = w * j3;
                    jacobians[0][i * (*param_blocks_sizes_)[0] + 3] = w * j4;
                    jacobians[0][i * (*param_blocks_sizes_)[0] + 4] = w * j5;
                    jacobians[0][i * (*param_blocks_sizes_)[0] + 5] = w * j6;
                    jacobians[0][i * (*param_blocks_sizes_)[0] + 6] = w * j7;
                }

                if(jacobians[1] != NULL)
                {
                    float j8, j9, j10, j11, j12, j13, j14;

                    float t74 = qrh*qrh;
                    float t75 = qxh*qxh;
                    float t76 = qyh*qyh;
                    float t77 = qzh*qzh;
                    float t78 = qrk*qrk;
                    float t79 = qxk*qxk;
                    float t80 = qyk*qyk;
                    float t81 = qzk*qzk;
                    float t82 = t78+t79+t80+t81;
                    float t83 = 1.0/t82;
                    float t84 = t74+t75+t76+t77;
                    float t85 = 1.0/t84;
                    float t86 = qrh*qzh*2.0;
                    float t93 = qxh*qyh*2.0;
                    float t87 = t86-t93;
                    float t88 = t74+t75-t76-t77;
                    float t89 = 1.0/(t82*t82);
                    float t90 = qrh*qyh*2.0;
                    float t91 = qxh*qzh*2.0;
                    float t92 = t90+t91;
                    float t94 = t74-t75+t76-t77;
                    float t95 = t78-t79+t80-t81;
                    float t96 = qrh*qxh*2.0;
                    float t104 = qyh*qzh*2.0;
                    float t97 = -t104+t96;
                    float t98 = qrk*qxk*2.0;
                    float t106 = qyk*qzk*2.0;
                    float t99 = -t106+t98;
                    float t100 = t86+t93;
                    float t101 = qrk*qzk*2.0;
                    float t102 = qxk*qyk*2.0;
                    float t103 = t101+t102;
                    float t105 = t104+t96;
                    float t107 = t74-t75-t76+t77;
                    float t108 = t90-t91;
                    float t109 = t78-t79-t80+t81;
                    float t110 = t106+t98;
                    float t111 = qrk*qyk*2.0;
                    float t113 = qxk*qzk*2.0;
                    float t112 = t111-t113;
                    float t114 = t78+t79-t80-t81;
                    float t115 = t101-t102;
                    float t116 = t111+t113;
                    float t117 = qrk*tzk*2.0;
                    float t118 = qxk*tyk*2.0;
                    float t193 = qyk*txk*2.0;
                    float t119 = t117+t118-t193;
                    float t120 = qyk*t83*t85*t88*2.0;
                    float t121 = qxk*t83*t85*t87*2.0;
                    float t122 = qxk*t83*t85*t94*2.0;
                    float t123 = qrk*t107*t83*t85*2.0;
                    float t124 = qxk*t105*t83*t85*2.0;
                    float t125 = qyk*t108*t83*t85*2.0;
                    float t126 = qrk*t83*tzh*2.0;
                    float t127 = qxk*t83*tyh*2.0;
                    float t128 = t78*txk;
                    float t129 = t79*txk;
                    float t130 = qrk*qyk*tzk*2.0;
                    float t131 = qxk*qyk*tyk*2.0;
                    float t132 = qxk*qzk*tzk*2.0;
                    float t170 = t80*txk;
                    float t171 = t81*txk;
                    float t172 = qrk*qzk*tyk*2.0;
                    float t133 = t128+t129+t130+t131+t132-t170-t171-t172;
                    float t134 = t78*tzk;
                    float t135 = t81*tzk;
                    float t136 = qrk*qxk*tyk*2.0;
                    float t137 = qxk*qzk*txk*2.0;
                    float t138 = qyk*qzk*tyk*2.0;
                    float t190 = t79*tzk;
                    float t191 = t80*tzk;
                    float t192 = qrk*qyk*txk*2.0;
                    float t139 = t134+t135+t136+t137+t138-t190-t191-t192;
                    float t140 = qxk*txk*2.0;
                    float t141 = qyk*tyk*2.0;
                    float t142 = qzk*tzk*2.0;
                    float t143 = t140+t141+t142;
                    float t144 = qyk*t83*t85*t87*2.0;
                    float t145 = qyk*t83*t85*t94*2.0;
                    float t146 = qxk*t100*t83*t85*2.0;
                    float t147 = qzk*t107*t83*t85*2.0;
                    float t148 = qyk*t105*t83*t85*2.0;
                    float t149 = qxk*t83*txh*2.0;
                    float t150 = qyk*t83*tyh*2.0;
                    float t151 = qzk*t83*tzh*2.0;
                    float t152 = t78*tyk;
                    float t153 = t80*tyk;
                    float t154 = qrk*qzk*txk*2.0;
                    float t155 = qxk*qyk*txk*2.0;
                    float t156 = qyk*qzk*tzk*2.0;
                    float t184 = t79*tyk;
                    float t185 = t81*tyk;
                    float t186 = qrk*qxk*tzk*2.0;
                    float t157 = t152+t153+t154+t155+t156-t184-t185-t186;
                    float t158 = qrk*tyk*2.0;
                    float t159 = qzk*txk*2.0;
                    float t194 = qxk*tzk*2.0;
                    float t160 = t158+t159-t194;
                    float t161 = t160*t83;
                    float t162 = qrk*t83*t85*t87*2.0;
                    float t163 = qxk*t83*t85*t92*2.0;
                    float t164 = qrk*t83*t85*t94*2.0;
                    float t165 = qxk*t83*t85*t97*2.0;
                    float t166 = qzk*t100*t83*t85*2.0;
                    float t167 = qxk*t107*t83*t85*2.0;
                    float t168 = qzk*t108*t83*t85*2.0;
                    float t169 = qxk*t83*tzh*2.0;
                    float t173 = qrk*txk*2.0;
                    float t174 = qyk*tzk*2.0;
                    float t198 = qzk*tyk*2.0;
                    float t175 = t173+t174-t198;
                    float t176 = qrk*t83*t85*t88*2.0;
                    float t177 = qyk*t83*t85*t92*2.0;
                    float t178 = qzk*t83*t85*t87*2.0;
                    float t179 = qzk*t83*t85*t94*2.0;
                    float t180 = qyk*t83*t85*t97*2.0;
                    float t181 = qyk*t107*t83*t85*2.0;
                    float t182 = qrk*t83*txh*2.0;
                    float t183 = qyk*t83*tzh*2.0;
                    float t187 = qxk*t83*t85*t88*2.0;
                    float t188 = qzk*t83*t85*t92*2.0;
                    float t189 = qzk*t83*t85*t97*2.0;
                    float t195 = qzk*t83*t85*t88*2.0;
                    float t196 = qrk*t83*tyh*2.0;
                    float t197 = qzk*t83*txh*2.0;

                    j8 = Lx*(t149+t150+t151+y*(t145+t146-qzk*t83*t85*t97*2.0-qxk*t100*t114*t85*t89*2.0+qxk*t115*t85*t89*t94*2.0+qxk*t116*t85*t89*t97*2.0)+z*(t147+t148-qxk*t108*t83*t85*2.0+qxk*t105*t115*t85*t89*2.0+qxk*t108*t114*t85*t89*2.0-qxk*t107*t116*t85*t89*2.0)-t143*t83-x*(t144-qxk*t83*t85*t88*2.0-qzk*t83*t85*t92*2.0+qxk*t115*t85*t87*t89*2.0+qxk*t114*t85*t88*t89*2.0+qxk*t116*t85*t89*t92*2.0)+qxk*t133*t89*2.0-qxk*t114*t89*txh*2.0+qxk*t115*t89*tyh*2.0-qxk*t116*t89*tzh*2.0)-Lz*(t161+t169+x*(t162+t163-qzk*t83*t85*t88*2.0-qxk*t110*t85*t87*t89*2.0-qxk*t112*t85*t88*t89*2.0+qxk*t109*t85*t89*t92*2.0)+z*(t167+t168-qrk*t105*t83*t85*2.0+qxk*t107*t109*t85*t89*2.0+qxk*t105*t110*t85*t89*2.0+qxk*t108*t112*t85*t89*2.0)-y*(t164+t165+t166+qxk*t100*t112*t85*t89*2.0-qxk*t110*t85*t89*t94*2.0+qxk*t109*t85*t89*t97*2.0)-qxk*t139*t89*2.0-qrk*t83*tyh*2.0-qzk*t83*txh*2.0-qxk*t112*t89*txh*2.0+qxk*t110*t89*tyh*2.0+qxk*t109*t89*tzh*2.0)-Ly*(t126+t127-x*(t120+t121-qrk*t83*t85*t92*2.0-qxk*t103*t85*t88*t89*2.0+qxk*t85*t87*t89*t95*2.0+qxk*t85*t89*t92*t99*2.0)-t119*t83+z*(t123+t124+t125-qxk*t103*t108*t85*t89*2.0+qxk*t105*t85*t89*t95*2.0-qxk*t107*t85*t89*t99*2.0)+y*(t122-qyk*t100*t83*t85*2.0-qrk*t83*t85*t97*2.0+qxk*t100*t103*t85*t89*2.0+qxk*t85*t89*t94*t95*2.0+qxk*t85*t89*t97*t99*2.0)-qxk*t157*t89*2.0-qyk*t83*txh*2.0+qxk*t103*t89*txh*2.0+qxk*t89*t95*tyh*2.0-qxk*t89*t99*tzh*2.0);
                    j9 = Ly*(t149+t150+t151+z*(t147+t148-qxk*t108*t83*t85*2.0+qyk*t103*t108*t85*t89*2.0-qyk*t105*t85*t89*t95*2.0+qyk*t107*t85*t89*t99*2.0)-y*(-t145-t146+t189+qyk*t100*t103*t85*t89*2.0+qyk*t85*t89*t94*t95*2.0+qyk*t85*t89*t97*t99*2.0)-t143*t83+x*(-t144+t187+t188-qyk*t103*t85*t88*t89*2.0+qyk*t85*t87*t89*t95*2.0+qyk*t85*t89*t92*t99*2.0)+qyk*t157*t89*2.0-qyk*t103*t89*txh*2.0-qyk*t89*t95*tyh*2.0+qyk*t89*t99*tzh*2.0)-Lz*(t182+t183-y*(t179+t180-qrk*t100*t83*t85*2.0+qyk*t100*t112*t85*t89*2.0-qyk*t110*t85*t89*t94*2.0+qyk*t109*t85*t89*t97*2.0)-t175*t83+x*(t176+t177+t178-qyk*t110*t85*t87*t89*2.0-qyk*t112*t85*t88*t89*2.0+qyk*t109*t85*t89*t92*2.0)+z*(t181-qrk*t108*t83*t85*2.0-qzk*t105*t83*t85*2.0+qyk*t107*t109*t85*t89*2.0+qyk*t105*t110*t85*t89*2.0+qyk*t108*t112*t85*t89*2.0)-qyk*t139*t89*2.0-qzk*t83*tyh*2.0-qyk*t112*t89*txh*2.0+qyk*t110*t89*tyh*2.0+qyk*t109*t89*tzh*2.0)+Lx*(t126+t127-x*(t120+t121-qrk*t83*t85*t92*2.0+qyk*t115*t85*t87*t89*2.0+qyk*t114*t85*t88*t89*2.0+qyk*t116*t85*t89*t92*2.0)-t119*t83+z*(t123+t124+t125+qyk*t105*t115*t85*t89*2.0+qyk*t108*t114*t85*t89*2.0-qyk*t107*t116*t85*t89*2.0)+y*(t122-qyk*t100*t83*t85*2.0-qrk*t83*t85*t97*2.0-qyk*t100*t114*t85*t89*2.0+qyk*t115*t85*t89*t94*2.0+qyk*t116*t85*t89*t97*2.0)+qyk*t133*t89*2.0-qyk*t83*txh*2.0-qyk*t114*t89*txh*2.0+qyk*t115*t89*tyh*2.0-qyk*t116*t89*tzh*2.0);
                    j10 = Lz*(t149+t150+t151-t143*t83+x*(-t144+t187+t188+qzk*t110*t85*t87*t89*2.0+qzk*t112*t85*t88*t89*2.0-qzk*t109*t85*t89*t92*2.0)+y*(t145+t146-t189+qzk*t100*t112*t85*t89*2.0-qzk*t110*t85*t89*t94*2.0+qzk*t109*t85*t89*t97*2.0)-z*(-t147-t148+qxk*t108*t83*t85*2.0+qzk*t107*t109*t85*t89*2.0+qzk*t105*t110*t85*t89*2.0+qzk*t108*t112*t85*t89*2.0)+qzk*t139*t89*2.0+qzk*t112*t89*txh*2.0-qzk*t110*t89*tyh*2.0-qzk*t109*t89*tzh*2.0)-Lx*(-t161-t169+t196+t197-z*(t167+t168-qrk*t105*t83*t85*2.0+qzk*t105*t115*t85*t89*2.0+qzk*t108*t114*t85*t89*2.0-qzk*t107*t116*t85*t89*2.0)+x*(-t162-t163+t195+qzk*t115*t85*t87*t89*2.0+qzk*t114*t85*t88*t89*2.0+qzk*t116*t85*t89*t92*2.0)+y*(t164+t165+t166+qzk*t100*t114*t85*t89*2.0-qzk*t115*t85*t89*t94*2.0-qzk*t116*t85*t89*t97*2.0)-qzk*t133*t89*2.0+qzk*t114*t89*txh*2.0-qzk*t115*t89*tyh*2.0+qzk*t116*t89*tzh*2.0)+Ly*(t182+t183-y*(t179+t180-qrk*t100*t83*t85*2.0+qzk*t100*t103*t85*t89*2.0+qzk*t85*t89*t94*t95*2.0+qzk*t85*t89*t97*t99*2.0)-t175*t83+x*(t176+t177+t178-qzk*t103*t85*t88*t89*2.0+qzk*t85*t87*t89*t95*2.0+qzk*t85*t89*t92*t99*2.0)+z*(t181-qrk*t108*t83*t85*2.0-qzk*t105*t83*t85*2.0+qzk*t103*t108*t85*t89*2.0-qzk*t105*t85*t89*t95*2.0+qzk*t107*t85*t89*t99*2.0)+qzk*t157*t89*2.0-qzk*t83*tyh*2.0-qzk*t103*t89*txh*2.0-qzk*t89*t95*tyh*2.0+qzk*t89*t99*tzh*2.0);
                    j11 = -Ly*(t161+t169-t196-t197+z*(t167+t168-qrk*t105*t83*t85*2.0-qrk*t103*t108*t85*t89*2.0+qrk*t105*t85*t89*t95*2.0-qrk*t107*t85*t89*t99*2.0)-y*(t164+t165+t166-qrk*t100*t103*t85*t89*2.0-qrk*t85*t89*t94*t95*2.0-qrk*t85*t89*t97*t99*2.0)+x*(t162+t163-t195+qrk*t103*t85*t88*t89*2.0-qrk*t85*t87*t89*t95*2.0-qrk*t85*t89*t92*t99*2.0)-qrk*t157*t89*2.0+qrk*t103*t89*txh*2.0+qrk*t89*t95*tyh*2.0-qrk*t89*t99*tzh*2.0)+Lx*(t182+t183-y*(t179+t180-qrk*t100*t83*t85*2.0+qrk*t100*t114*t85*t89*2.0-qrk*t115*t85*t89*t94*2.0-qrk*t116*t85*t89*t97*2.0)-t175*t83+x*(t176+t177+t178-qrk*t115*t85*t87*t89*2.0-qrk*t114*t85*t88*t89*2.0-qrk*t116*t85*t89*t92*2.0)+z*(t181-qrk*t108*t83*t85*2.0-qzk*t105*t83*t85*2.0+qrk*t105*t115*t85*t89*2.0+qrk*t108*t114*t85*t89*2.0-qrk*t107*t116*t85*t89*2.0)+qrk*t133*t89*2.0-qzk*t83*tyh*2.0-qrk*t114*t89*txh*2.0+qrk*t115*t89*tyh*2.0-qrk*t116*t89*tzh*2.0)+Lz*(t126+t127-x*(t120+t121-qrk*t83*t85*t92*2.0-qrk*t110*t85*t87*t89*2.0-qrk*t112*t85*t88*t89*2.0+qrk*t109*t85*t89*t92*2.0)-t119*t83+z*(t123+t124+t125-qrk*t107*t109*t85*t89*2.0-qrk*t105*t110*t85*t89*2.0-qrk*t108*t112*t85*t89*2.0)+y*(t122-qyk*t100*t83*t85*2.0-qrk*t83*t85*t97*2.0+qrk*t100*t112*t85*t89*2.0-qrk*t110*t85*t89*t94*2.0+qrk*t109*t85*t89*t97*2.0)+qrk*t139*t89*2.0-qyk*t83*txh*2.0+qrk*t112*t89*txh*2.0-qrk*t110*t89*tyh*2.0-qrk*t109*t89*tzh*2.0);
                    j12 = -Lx*t114*t83-Ly*t103*t83+Lz*t112*t83;
                    j13 = Lx*t115*t83-Lz*t110*t83-Ly*t83*t95;
                    j14 = -Lx*t116*t83-Lz*t109*t83+Ly*t83*t99;

                    jacobians[1][i * (*param_blocks_sizes_)[1]    ] = w * j8;
                    jacobians[1][i * (*param_blocks_sizes_)[1] + 1] = w * j9;
                    jacobians[1][i * (*param_blocks_sizes_)[1] + 2] = w * j10;
                    jacobians[1][i * (*param_blocks_sizes_)[1] + 3] = w * j11;
                    jacobians[1][i * (*param_blocks_sizes_)[1] + 4] = w * j12;
                    jacobians[1][i * (*param_blocks_sizes_)[1] + 5] = w * j13;
                    jacobians[1][i * (*param_blocks_sizes_)[1] + 6] = w * j14;
                }
            }
        }

        /*if(jacobians != NULL)
        {
            if(jacobians[0] != NULL)
            {
                std::cout << sizeof(jacobians[0]) << std::endl;
            }
        }*/


        /*for(size_t i=0; i < nl_icp->clouds_transformed_with_ip_[h_]->points.size(); i++)
        {
            if(jacobians != NULL)
            {
                if(jacobians[0] != NULL)
                {
                    std::cout << "Parameter block 0:" << std::endl;
                    for(size_t k=0; k < (*param_blocks_sizes_)[0]; k++)
                    {
                        std::cout << jacobians[0][i * (*param_blocks_sizes_)[0] + k] << " ";
                    }

                    std::cout << std::endl;
                }

                if(jacobians[1] != NULL)
                {
                    std::cout << "Parameter block 1:" << std::endl;
                    for(size_t k=0; k < (*param_blocks_sizes_)[1]; k++)
                    {
                        std::cout << jacobians[1][i * (*param_blocks_sizes_)[1] + k] << " ";
                    }

                    std::cout << std::endl;
                }
            }
        }*/

        return true;
    }
};

template<class PointT>
v4r::Registration::MvLMIcp<PointT>::MvLMIcp()
{
    max_correspondence_distance_ = 0.01;
    max_iterations_ = 5;
    diff_type = 2;
    normal_dot_ = 0.9f;
}

template<class PointT>
void
v4r::Registration::MvLMIcp<PointT>::compute()
{
    //compute octrees for all point clouds
    //used for adjacency matrix computation as well as during the optimization (except when DT are used)

    clouds_transformed_with_ip_.resize(clouds_.size());

    if(normals_.size() == clouds_.size())
        normals_transformed_with_ip_.resize(clouds_.size());

    for (size_t i = 0; i < clouds_.size (); i++)
    {
        clouds_transformed_with_ip_[i].reset(new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*clouds_[i], *clouds_transformed_with_ip_[i], poses_[i]);

        if(normals_.size() == clouds_.size())
        {
            normals_transformed_with_ip_[i].reset(new pcl::PointCloud<pcl::Normal>());
            v4r::transformNormals(*normals_[i], *normals_transformed_with_ip_[i], poses_[i]);
        }
    }

    if(diff_type == 1)
    {
        //using distance transforms
        distance_transforms_.resize(clouds_.size());

        for (size_t i = 0; i < clouds_.size (); i++)
        {
            typename pcl::PointCloud<PointT>::ConstPtr cloud_const(new pcl::PointCloud<PointT>(*clouds_transformed_with_ip_[i]));
            distance_transforms_[i].reset(new distance_field::PropagationDistanceField<PointT>(0.003));
            distance_transforms_[i]->setInputCloud(cloud_const);
            distance_transforms_[i]->compute();
            distance_transforms_[i]->computeFiniteDifferences();
        }
    }
    else
    {
        octrees_.resize(clouds_.size());

        for (size_t i = 0; i < clouds_.size (); i++)
        {
            octrees_[i].reset(new pcl::octree::OctreePointCloudSearch<PointT> (0.003));
            octrees_[i]->setInputCloud (clouds_transformed_with_ip_[i]);
            octrees_[i]->addPointsFromInputCloud ();
        }

    }

    //fill adjacency matrix and view pair list
    computeAdjacencyMatrix();
    fillViewParList();

    std::cout << "view pairs used in registration:" << S_.size() << std::endl;

    //optimize :)
    //note: the jacobian matrix is cardinality(S_) * [ sizeof(clouds_) * { sizeof(pose) = 6 } ]
    //      it is a block-row matrix, for a block-row s, everything is zero except the rate of change for
    //      for the parameters associated with s.first and s.second
    //      it is a sparse-matrix

    ceres::Problem problem;

    //int diff_type = 2; //0-numeric diff, 1-mixed, 2-analytic

    std::cout << "DIFF TYPE:" << diff_type << std::endl;

    int params_per_view = 6;
    double * parameters;

    switch(diff_type)
    {
    case 0:
    {

        parameters = new double[params_per_view * clouds_.size()];
        for(size_t i=0; i < clouds_.size(); i++)
        {
            //rotation (rx,ry,rz)
            parameters[i * params_per_view + 0] = 0.0;
            parameters[i * params_per_view + 1] = 0.0;
            parameters[i * params_per_view + 2] = 0.0;
            //translation (tx,ty,tz)
            parameters[i * params_per_view + 3] = 0.0;
            parameters[i * params_per_view + 4] = 0.0;
            parameters[i * params_per_view + 5] = 0.0;
        }

        break;
    }

    case 1:
    {
        parameters = new double[params_per_view * clouds_.size()];
        for(size_t i=0; i < clouds_.size(); i++)
        {
            //rotation (rx,ry,rz)
            parameters[i * params_per_view + 0] = 0.0;
            parameters[i * params_per_view + 1] = 0.0;
            parameters[i * params_per_view + 2] = 0.0;
            //translation (tx,ty,tz)
            parameters[i * params_per_view + 3] = 0.0;
            parameters[i * params_per_view + 4] = 0.0;
            parameters[i * params_per_view + 5] = 0.0;
        }

        break;
    }

    default:
    {

        params_per_view = 7;
        parameters = new double[params_per_view * clouds_.size()];
        for(size_t i=0; i < clouds_.size(); i++)
        {
            //rotation (qx,qy,qz,qw)
            parameters[i * params_per_view + 0] = 0.0;
            parameters[i * params_per_view + 1] = 0.0;
            parameters[i * params_per_view + 2] = 0.0;
            parameters[i * params_per_view + 3] = 1.0;
            //translation (tx,ty,tz)
            parameters[i * params_per_view + 4] = 0.0;
            parameters[i * params_per_view + 5] = 0.0;
            parameters[i * params_per_view + 6] = 0.0;
        }

        break;
    }
    }

    //how many residual blocks do I need to add?
    //for each pair of views involved in the registration, add a residual block
    for(size_t i=0; i < S_.size(); i++)
    {

        switch(diff_type)
        {
        case 0:
        {

            ceres::CostFunction* cost_function
                    = new ceres::NumericDiffCostFunction<RegistrationCostFunctor<PointT>, ceres::CENTRAL, ceres::DYNAMIC, 6, 6>(
                        new RegistrationCostFunctor<PointT>(this, S_[i].first, S_[i].second),
                        ceres::TAKE_OWNERSHIP,
                        static_cast<int>(clouds_[S_[i].first]->points.size()));

            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(max_correspondence_distance_ / 2.0),
                                     parameters + S_[i].first * params_per_view,
                                     parameters + S_[i].second * params_per_view);
            break;
        }

        case 1:
        {

            RegistrationCostFunctionAutoDiff<PointT> * functor
                    = new RegistrationCostFunctionAutoDiff<PointT> (this, S_[i].first, S_[i].second);

            ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction< RegistrationCostFunctionAutoDiff<PointT>, ceres::DYNAMIC, 6, 6>
                    (functor, static_cast<int>(clouds_[S_[i].first]->points.size()));

            problem.AddResidualBlock(cost_function,
                                     new ceres::CauchyLoss(max_correspondence_distance_ / 2.0),
                                     parameters + S_[i].first * params_per_view,
                                     parameters + S_[i].second * params_per_view);

            break;
        }

        default:
        {

            ceres::CostFunction* cost_function = new RegistrationCostFunction<PointT>(this, S_[i].first, S_[i].second);

            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(max_correspondence_distance_ / 2.0),
                                     parameters + S_[i].first * params_per_view,
                                     parameters + S_[i].second * params_per_view);

            break;
        }
        }
    }

    problem.SetParameterBlockConstant(parameters);

    // Run the solver!
    bool verbose_ = true;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = verbose_;
    options.check_gradients = false;
    options.max_num_iterations = max_iterations_;
    options.function_tolerance = 1e-6;
    options.num_threads = 4;
    options.num_linear_solver_threads = 4;
    options.numeric_derivative_relative_step_size = 1e-8;
    if(diff_type == 2)
        options.initial_trust_region_radius = 1e-2;

    //options.
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if(verbose_)
    {
        std::cout << summary.FullReport() << "\n";
    }

    final_poses_.clear();
    final_poses_.resize(poses_.size());
    for(size_t i=0; i < clouds_.size(); i++)
    {

        switch(diff_type)
        {
        case 0:
        {

            Eigen::Vector3d trans(parameters[i * params_per_view + 3], parameters[i * params_per_view + 4], parameters[i * params_per_view + 5]);
            Eigen::Matrix3d R;
            ceres::AngleAxisToRotationMatrix<double>(parameters + i * params_per_view, R.data());

            std::cout << R << std::endl;

            Eigen::Matrix4f T_h;
            T_h.setIdentity();
            T_h.block<3,3>(0,0) = R.cast<float>();
            T_h.block<3,1>(0,3) = trans.cast<float>();
            final_poses_[i] = (T_h * poses_[i]);

            break;
        }

        case 1:
        {

            Eigen::Vector3d trans(parameters[i * params_per_view + 3], parameters[i * params_per_view + 4], parameters[i * params_per_view + 5]);
            Eigen::Matrix3d R;
            ceres::AngleAxisToRotationMatrix<double>(parameters + i * params_per_view, R.data());

            std::cout << R << std::endl;

            Eigen::Matrix4f T_h;
            T_h.setIdentity();
            T_h.block<3,3>(0,0) = R.cast<float>();
            T_h.block<3,1>(0,3) = trans.cast<float>();
            final_poses_[i] = (T_h * poses_[i]);

            break;
        }

        default:
        {

            Eigen::Quaterniond rot(parameters[i * params_per_view + 3], parameters[i * params_per_view], parameters[i * params_per_view + 1], parameters[i * params_per_view + 2]);
            Eigen::Vector3d trans(parameters[i * params_per_view + 4], parameters[i * params_per_view + 5], parameters[i * params_per_view + 6]);
            Eigen::Matrix3d R = coolquat_to_mat(rot);
            R /= rot.squaredNorm();

            Eigen::Matrix4f T_h;
            T_h.setIdentity();
            T_h.block<3,3>(0,0) = R.cast<float>();
            T_h.block<3,1>(0,3) = trans.cast<float>();
            final_poses_[i] = (T_h * poses_[i]);


            break;
        }
        }

        for(int k=0; k < params_per_view; k++)
        {
            std::cout << parameters[i*params_per_view + k] << " ";
        }

        std::cout << std::endl;

    }
}

template<class PointT>
void v4r::Registration::MvLMIcp<PointT>::computeAdjacencyMatrix()
{
    adjacency_matrix_.resize(clouds_.size());
    for(size_t i=0; i < clouds_.size(); i++)
    {
        adjacency_matrix_[i].resize(clouds_.size(), false);
    }

    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;
    float inlier = max_correspondence_distance_ * 2.f;
    for (size_t i = 0; i < clouds_.size (); i++)
    {

        for (size_t j = (i+1); j < clouds_.size (); j++)
        {
            //compute overlap
            int overlap = 0;
            for (size_t kk = 0; kk < clouds_[j]->points.size (); kk++)
            {
                if(pcl_isnan(clouds_transformed_with_ip_[j]->points[kk].x))
                    continue;

                if(diff_type == 1)
                {

                    float dist;
                    int idx;
                    distance_transforms_[i]->getCorrespondence(clouds_transformed_with_ip_[j]->points[kk], &idx, &dist, 0, 0);
                    if (dist < inlier)
                    {
                        overlap++;
                    }
                }
                else
                {
                    if (octrees_[i]->nearestKSearch (clouds_transformed_with_ip_[j]->points[kk], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        float d = sqrt (pointNKNSquaredDistance[0]);
                        if (d < inlier)
                        {
                            overlap++;
                        }
                    }
                }
            }

            float ov_measure_1 = overlap / static_cast<float>(clouds_[j]->points.size());
            float ov_measure_2 = overlap / static_cast<float>(clouds_[i]->points.size());
            float ff = 0.3f;
            if(ov_measure_1 > ff || ov_measure_2 > ff)
            {
                adjacency_matrix_[i][j] = adjacency_matrix_[j][i] = true;
            }
        }
    }

    for (size_t i = 0; i < adjacency_matrix_.size (); i++)
    {
        for (size_t j = 0; j < adjacency_matrix_.size (); j++)
            std::cout << adjacency_matrix_[i][j] << " ";

        std::cout << std::endl;
    }
}

template<class PointT>
void
v4r::Registration::MvLMIcp<PointT>::fillViewParList()
{
    for(size_t i=0; i < clouds_.size(); i++)
    {
        for(size_t j=(i+1); j < clouds_.size(); j++)
        {
            if(adjacency_matrix_[i][j])
            {
                std::pair<int, int> p = std::make_pair<int, int>((int)i,(int)j);
                S_.push_back(p);
            }
        }
    }
}


template class V4R_EXPORTS v4r::Registration::MvLMIcp<pcl::PointXYZRGB>;

