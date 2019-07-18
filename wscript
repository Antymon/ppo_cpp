#! /usr/bin/env python
import sys
import os
import sferes
sys.path.insert(0, sys.path[0]+'/waf_tools')
print sys.path[0]


from waflib.Configure import conf

def options(opt):
    opt.load('dart')

@conf
def configure(conf):
    print 'conf exp:'
    conf.load('dart')
    conf.check_dart()
    print 'done'
    
def build(bld):
    # bld.env.LIBPATH_ABSL = '/workspace/lib'
    # bld.env.LIB_ABSL = ['absl_algorithm', 'absl_any', 'absl_bad_any_cast', 'absl_bad_optional_access', 'absl_base', 'absl_base_internal_exception_safety_testing', 'absl_container', 'absl_debugging', 'absl_dynamic_annotations', 'absl_examine_stack', 'absl_failure_signal_handler', 'absl_int128', 'absl_leak_check', 'absl_malloc_internal', 'absl_memory', 'absl_meta', 'absl_numeric', 'absl_optional', 'absl_span', 'absl_spinlock_wait', 'absl_stack_consumption', 'absl_stacktrace', 'absl_strings', 'absl_symbolize', 'absl_synchronization', 'absl_throw_delegate', 'absl_time', 'absl_utility', 'absl_variant']
    # bld.env.INCLUDES_ABSL = '/workspace/include/absl/'

    bld.env.LIBPATH_ROBOTDART = '/workspace/lib'
    bld.env.LIB_ROBOTDART = ['RobotDARTSimu']
    bld.env.INCLUDES_ROBOTDART = '/workspace/include/robot_dart/src'

    bld.env.LIBPATH_TF = '/workspace/lib'
    bld.env.LIB_TF = [ 'tensorflow_cc', 'tensorflow_framework']
    bld.env.INCLUDES_TF = '/workspace/include/google/tensorflow/'

    bld.env.LIBPATH_PROTOBUF = '/workspace/lib'
    bld.env.LIB_PROTOBUF = [ 'protobuf', 'protoc']
    bld.env.INCLUDES_PROTOBUF = '/workspace/include/'

    bld.program(features = 'cxx',
            #source = 'cpp/tf_exp.cpp',
            source = 'ppo2.cpp',
            includes = './cpp . ../../',
            uselib = 'ROBOTDART ABSL TBB BOOST EIGEN PTHREAD MPI DART DART_GRAPHIC PROTOBUF TF',
            #use = 'sferes2',
            defines = ['GRAPHIC'],
            target = 'ppo_cpp')