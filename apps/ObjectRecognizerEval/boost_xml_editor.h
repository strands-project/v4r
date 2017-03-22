#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

struct XMLChange
{
    const std::string xml_filename_;
    const std::string node_name_;
    const std::vector<std::string> values_;
    bool is_multitoken_;
    const std::string tmp_xml_filename_;

    XMLChange(
            const std::string &xml_filename,
            const std::string &node_name,
            const std::vector<std::string> &values = std::vector<std::string>(),
            bool is_multitoken = 0,
            const std::string &tmp_xml_filename = "/tmp/out.xml"
            )
        :
          xml_filename_ ( xml_filename ),
          node_name_ ( node_name ),
          values_ ( values ),
          is_multitoken_ (is_multitoken),
          tmp_xml_filename_ ( tmp_xml_filename )
    {  }
};

int
editXML(const std::string &xml_filename, const std::string &node_name, const std::vector<std::string> &values, bool is_multitoken = false, const std::string &tmp_xml_filename = "/tmp/out.xml")
{
    std::ifstream xml_f (xml_filename);
    std::ofstream xml_tmp (tmp_xml_filename);

    const std::string query_pattern = "<" + node_name + ">";

    bool found = false;

    std::string line;
    while (std::getline(xml_f, line))
    {
        size_t pos = line.find(query_pattern);
        if( pos != std::string::npos )
        {
            found = true;
            xml_tmp << line.substr(0, pos);
            if(!is_multitoken)
            {
                xml_tmp << "<" << node_name << ">" << values[0] << "</" << node_name << ">" << std::endl;
            }
            else
            {
                xml_tmp << line << std::endl;

                //get all count value to know how much to delete
                int count;
                std::getline(xml_f, line);
                int delimiter = line.find('>');
                count = std::stoi( line.substr( delimiter + 1, line.find('</') - delimiter - 2));
                xml_tmp << "<count>" << values.size() << "</count>" << std::endl;

                // ignore next (version) line
                std::getline(xml_f, line);
                xml_tmp << line << std::endl;

                // remove lines first
                for(int i=0; i<count; i++)
                    std::getline(xml_f, line);

                // now add new entries
                for(const std::string &val : values)
                {
                    xml_tmp << "<item>" << val << "</item>" << std::endl;
                }

            }
        }
        else
            xml_tmp << line << std::endl;
    }
    xml_f.close();
    xml_tmp.close();

    // now write back to original file
    if(found)
    {
        std::ifstream xml_tmp ("/tmp/out.xml");
        std::ofstream xml_f (xml_filename);

        while (std::getline(xml_tmp, line))
            xml_f << line << std::endl;

        xml_f.close();
        xml_tmp.close();
    }
    else
    {
        std::cerr << "DID NOT FIND ENTRY " << node_name << " in " << xml_filename << "!" << std::endl;
        return 0;
    }

    return 1;
}


int
editXML(const XMLChange &xml_change)
{
    return editXML( xml_change.xml_filename_, xml_change.node_name_, xml_change.values_, xml_change.is_multitoken_, xml_change.tmp_xml_filename_ );
}

std::string
getValue(const std::string &xml_filename, const std::string &node_name)
{
    std::ifstream xml_f (xml_filename);

    const std::string query_pattern = "<" + node_name + ">";

    bool found = false;

    std::string line;

    std::string value;
    while (std::getline(xml_f, line))
    {
        size_t pos = line.find(query_pattern);
        if( pos != std::string::npos )
        {
            found = true;
            int delimiter = line.find('>');
            value =  line.substr( delimiter + 1, line.find('</') - delimiter - 2);
            break;
        }
    }
    xml_f.close();

    // now write back to original file
    if(!found)
        std::cerr << "DID NOT FIND ENTRY " << node_name << " in " << xml_filename << "!" << std::endl;

    return value;
}

