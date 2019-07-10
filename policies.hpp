//
// Created by szymon on 10/07/19.
//

#ifndef PPO_CPP_POLICIES_HPP
#define PPO_CPP_POLICIES_HPP

class BasePolicy{

};

class ActorCriticPolicy : public virtual BasePolicy{

};

class FeedForwardPolicy : public virtual ActorCriticPolicy{

};

class MlpPolicy : public  virtual  FeedForwardPolicy{

};

#endif //PPO_CPP_POLICIES_HPP
