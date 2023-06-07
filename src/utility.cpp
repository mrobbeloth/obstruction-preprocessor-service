#include <iostream>
#include <filesystem>
#include<algorithm>
using namespace std;
using namespace filesystem;
std::size_t number_of_files_in_directory(std::filesystem::path path)
{
    using std::filesystem::directory_iterator;
    using fp = bool (*)( const std::filesystem::path&);
    return count_if(directory_iterator(path), directory_iterator{}, (fp)std::filesystem::is_regular_file);
}