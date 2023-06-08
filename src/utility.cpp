#include "utility.h"
std::size_t number_of_files_in_directory(std::filesystem::path path)
{
    using std::filesystem::directory_iterator;
    using fp = bool (*)( const std::filesystem::path&);
    return count_if(directory_iterator(path), directory_iterator{}, (fp)std::filesystem::is_regular_file);
}

std::vector<std::string> findFiles(const std::string& directory, 
                                   const std::string& extension)
{
    std::vector<std::string> pngFiles;
    
    for (const auto& entry : filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == extension)
        {
            pngFiles.push_back(entry.path().filename().string());
        }
    }
    
    return pngFiles;
}