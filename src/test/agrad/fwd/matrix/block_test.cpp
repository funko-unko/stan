#include <gtest/gtest.h>
#include <stan/math/matrix/block.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/var.hpp>

TEST(AgradFwdMatrix,block_matrix) {
  using stan::math::block;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_fv v(3,3);
  v << 1, 4, 9,1, 4, 9,1, 4, 9;
   v(0,0).d_ = 1.0;
   v(0,1).d_ = 2.0;
   v(0,2).d_ = 3.0;
   v(1,0).d_ = 1.0;
   v(1,1).d_ = 2.0;
   v(1,2).d_ = 3.0;
   v(2,0).d_ = 1.0;
   v(2,1).d_ = 2.0;
   v(2,2).d_ = 3.0;
  matrix_fv m = block(v, 1,1,3,3);
  EXPECT_EQ(1,m(0,0).val_);
  EXPECT_EQ(4,m(0,1).val_);
  EXPECT_EQ(9,m(0,2).val_);
  EXPECT_EQ(1,m(1,0).val_);
  EXPECT_EQ(4,m(1,1).val_);
  EXPECT_EQ(9,m(1,2).val_);
  EXPECT_EQ(1,m(2,0).val_);
  EXPECT_EQ(4,m(2,1).val_);
  EXPECT_EQ(9,m(2,2).val_);
  EXPECT_EQ(1,m(0,0).val_);
  EXPECT_EQ(2,m(0,1).d_);
  EXPECT_EQ(3,m(0,2).d_);
  EXPECT_EQ(1,m(1,0).d_);
  EXPECT_EQ(2,m(1,1).d_);
  EXPECT_EQ(3,m(1,2).d_);
  EXPECT_EQ(1,m(2,0).d_);
  EXPECT_EQ(2,m(2,1).d_);
  EXPECT_EQ(3,m(2,2).d_);

  matrix_fv n = block(v, 2,2,2,2);
  EXPECT_EQ(4,n(0,0).val_);
  EXPECT_EQ(9,n(0,1).val_);
  EXPECT_EQ(4,n(1,0).val_);
  EXPECT_EQ(9,n(1,1).val_);
  EXPECT_EQ(2,n(0,0).d_);
  EXPECT_EQ(3,n(0,1).d_);
  EXPECT_EQ(2,n(1,0).d_);
  EXPECT_EQ(3,n(1,1).d_);
}
TEST(AgradFwdMatrix,block_matrix_exception) {
  using stan::math::block;
  using stan::agrad::matrix_fv;

  matrix_fv v(3,3);
  EXPECT_THROW(block(v,0,0,1,1), std::domain_error);
  EXPECT_THROW(block(v,1,1,4,4), std::domain_error);
}
TEST(AgradFwdFvarVarMatrix,block_matrix) {
  using stan::math::block;
  using stan::agrad::matrix_fvv;
  using stan::agrad::vector_fvv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> a(1.0,1.0);
  fvar<var> b(4.0,2.0);
  fvar<var> c(9.0,3.0);
  matrix_fvv v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  matrix_fvv m = block(v, 1,1,3,3);
  EXPECT_EQ(1,m(0,0).val_.val());
  EXPECT_EQ(4,m(0,1).val_.val());
  EXPECT_EQ(9,m(0,2).val_.val());
  EXPECT_EQ(1,m(1,0).val_.val());
  EXPECT_EQ(4,m(1,1).val_.val());
  EXPECT_EQ(9,m(1,2).val_.val());
  EXPECT_EQ(1,m(2,0).val_.val());
  EXPECT_EQ(4,m(2,1).val_.val());
  EXPECT_EQ(9,m(2,2).val_.val());
  EXPECT_EQ(1,m(0,0).val_.val());
  EXPECT_EQ(2,m(0,1).d_.val());
  EXPECT_EQ(3,m(0,2).d_.val());
  EXPECT_EQ(1,m(1,0).d_.val());
  EXPECT_EQ(2,m(1,1).d_.val());
  EXPECT_EQ(3,m(1,2).d_.val());
  EXPECT_EQ(1,m(2,0).d_.val());
  EXPECT_EQ(2,m(2,1).d_.val());
  EXPECT_EQ(3,m(2,2).d_.val());

  matrix_fvv n = block(v, 2,2,2,2);
  EXPECT_EQ(4,n(0,0).val_.val());
  EXPECT_EQ(9,n(0,1).val_.val());
  EXPECT_EQ(4,n(1,0).val_.val());
  EXPECT_EQ(9,n(1,1).val_.val());
  EXPECT_EQ(2,n(0,0).d_.val());
  EXPECT_EQ(3,n(0,1).d_.val());
  EXPECT_EQ(2,n(1,0).d_.val());
  EXPECT_EQ(3,n(1,1).d_.val());
}
TEST(AgradFwdFvarVarMatrix,block_matrix_exception) {
  using stan::math::block;
  using stan::agrad::matrix_fvv;

  matrix_fvv v(3,3);
  EXPECT_THROW(block(v,0,0,1,1), std::domain_error);
  EXPECT_THROW(block(v,1,1,4,4), std::domain_error);
}
TEST(AgradFwdFvarFvarMatrix,block_matrix) {
  using stan::math::block;
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;
  using stan::agrad::fvar;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;

  a.val_.val_ = 1.0;
  a.d_.val_ = 1.0;  
  b.val_.val_ = 4.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 9.0;
  c.d_.val_ = 3.0;

  matrix_ffv v(3,3);
  v << a,b,c,a,b,c,a,b,c;

  matrix_ffv m = block(v, 1,1,3,3);
  EXPECT_EQ(1,m(0,0).val_.val());
  EXPECT_EQ(4,m(0,1).val_.val());
  EXPECT_EQ(9,m(0,2).val_.val());
  EXPECT_EQ(1,m(1,0).val_.val());
  EXPECT_EQ(4,m(1,1).val_.val());
  EXPECT_EQ(9,m(1,2).val_.val());
  EXPECT_EQ(1,m(2,0).val_.val());
  EXPECT_EQ(4,m(2,1).val_.val());
  EXPECT_EQ(9,m(2,2).val_.val());
  EXPECT_EQ(1,m(0,0).val_.val());
  EXPECT_EQ(2,m(0,1).d_.val());
  EXPECT_EQ(3,m(0,2).d_.val());
  EXPECT_EQ(1,m(1,0).d_.val());
  EXPECT_EQ(2,m(1,1).d_.val());
  EXPECT_EQ(3,m(1,2).d_.val());
  EXPECT_EQ(1,m(2,0).d_.val());
  EXPECT_EQ(2,m(2,1).d_.val());
  EXPECT_EQ(3,m(2,2).d_.val());

  matrix_ffv n = block(v, 2,2,2,2);
  EXPECT_EQ(4,n(0,0).val_.val());
  EXPECT_EQ(9,n(0,1).val_.val());
  EXPECT_EQ(4,n(1,0).val_.val());
  EXPECT_EQ(9,n(1,1).val_.val());
  EXPECT_EQ(2,n(0,0).d_.val());
  EXPECT_EQ(3,n(0,1).d_.val());
  EXPECT_EQ(2,n(1,0).d_.val());
  EXPECT_EQ(3,n(1,1).d_.val());
}
TEST(AgradFwdFvarFvarMatrix,block_matrix_exception) {
  using stan::math::block;
  using stan::agrad::matrix_ffv;

  matrix_ffv v(3,3);
  EXPECT_THROW(block(v,0,0,1,1), std::domain_error);
  EXPECT_THROW(block(v,1,1,4,4), std::domain_error);
}
