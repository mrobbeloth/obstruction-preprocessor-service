#include <iostream>
#include <filesystem>
#include<algorithm>
using namespace std;
using namespace filesystem;
std::size_t number_of_files_in_directory(std::filesystem::path path);
std::vector<std::string> findFiles(const std::string& directory, 
                                   const std::string& extension);