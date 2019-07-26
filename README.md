代码需要在tensorflow2.0上运行

输入支持letter_box操作

~~estimator训练方式loss为Nan，需要解决(已通过GN代替BN解决该问题，主要原因出现在BN)~~
keras模式可以正常训练,多gpu测试暂未通过，报段错误，预计和tf.data和strategy有一定关系
