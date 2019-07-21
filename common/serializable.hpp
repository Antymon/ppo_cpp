//
// Created by szymon on 21/07/19.
//

#ifndef PPO_CPP_SERIALIZABLE_HPP
#define PPO_CPP_SERIALIZABLE_HPP


#include "../json.hpp"

class ISerializable {
public:
    virtual void serialize(nlohmann::json& json) = 0;
    virtual void deserialize(nlohmann::json& json) = 0;
};


#endif //PPO_CPP_SERIALIZABLE_HPP
