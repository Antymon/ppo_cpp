//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_LOGGER_HPP
#define PPO_CPP_LOGGER_HPP

#include <string>
#include <iostream>

class Logger{
public:
    Logger(std::string name):
    _name{std::move(name)}{
        std::cout << _name << std::endl;
    }
    ~Logger(){
        std::cout << _name << " destructed" <<std::endl;
    }

    std::string _name;
};
#endif //PPO_CPP_LOGGER_HPP
