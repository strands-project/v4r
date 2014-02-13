#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <vtkPLYWriter.h>
#include <pcl/console/parse.h>
#include <fstream>

int
main (int argc, char ** argv)
{
  std::string vtk_file = "";
  std::string out_file = "";

  pcl::console::parse_argument (argc, argv, "-vtk_file", vtk_file);
  pcl::console::parse_argument (argc, argv, "-out_file", out_file);

  vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New ();
  reader->SetFileName (vtk_file.c_str());
  reader->Update ();
  vtkSmartPointer<vtkPolyData> polydata = reader->GetOutput ();

  vtkSmartPointer<vtkPLYWriter> writer = vtkSmartPointer<vtkPLYWriter>::New();
  writer->SetFileName(out_file.c_str());
  writer->SetInputConnection(reader->GetOutputPort());
  writer->Update();
  writer->Write();
}
