import tensorflow as tf

class REBNCONV (tf.keras.layers.Layer): 
    def __init__(self, out_ch = 3, dirate = 1, name = None):
        super (REBNCONV, self).__init__(name = name)
        self.conv_s1 = tf.keras.layers.Conv2D (filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = dirate)
        self.bn_s1 = tf.keras.layers.BatchNormalization()
    
    def __call__ (self, x, training=False):
        hx = x
        hx = self.conv_s1 (hx)
        hx = self.bn_s1 (hx,training)
        xout = tf.nn.relu (hx)
        return xout # return original size
    
## upsample tensor 'src' to have the same spatial size with tensor 'tar'
# suppose src and tar are tensor datadet
def _upsample_like(src,tar):
    src = tf.image.resize(src, size = tf.keras.backend.shape(tar)[1:3], method ='bilinear')
    return src


### RSU-7 ###
class RSU7 (tf.keras.layers.Layer): #UNet07DRES(nn.Module)
     def __init__(self, mid_ch = 12, out_ch = 3, name = None):
        super(RSU7,self).__init__(name = name)
        self.rebnconvin = REBNCONV(out_ch, dirate = 1) 

        self.rebnconv1 = REBNCONV(mid_ch, dirate = 1)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid') 

        self.rebnconv2 = REBNCONV(mid_ch, dirate = 1)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid') 

        self.rebnconv3 = REBNCONV(mid_ch, dirate = 1)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.rebnconv4 = REBNCONV(mid_ch, dirate = 1)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.rebnconv5 = REBNCONV(mid_ch, dirate = 1)
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.rebnconv6 = REBNCONV(mid_ch, dirate = 1)

        self.rebnconv7 = REBNCONV(mid_ch, dirate = 2)

        self.rebnconv6d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv5d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv4d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv3d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv1d = REBNCONV(out_ch, dirate = 1)
        
     def __call__ (self, x,training):
        hx = x
        hxin = self.rebnconvin(hx,training)

        hx1 = self.rebnconv1(hxin,training)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx,training)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx,training)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx,training)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx,training)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx,training)

        hx7 = self.rebnconv7(hx6,training)
        
        hx6d =  self.rebnconv6d(tf.concat([hx7,hx6],3),training) #okay
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(tf.concat([hx6dup,hx5],3),training)
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(tf.concat([hx5dup,hx4],3),training)
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(tf.concat([hx4dup,hx3],3),training)
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(tf.concat([hx3dup,hx2],3),training)
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(tf.concat([hx2dup,hx1],3),training)

        return hx1d + hxin #hx1d and hxin --> shapes are different 


### RSU-6 ###
class RSU6(tf.keras.layers.Layer):#UNet06DRES(nn.Module):
    def __init__(self, mid_ch = 12, out_ch = 3, name = None):
        super(RSU6,self).__init__(name = name)

        self.rebnconvin = REBNCONV(out_ch, dirate = 1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate = 1)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid') 

        self.rebnconv2 = REBNCONV(mid_ch, dirate = 1)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid') 

        self.rebnconv3 = REBNCONV(mid_ch, dirate = 1)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid') 

        self.rebnconv4 = REBNCONV(mid_ch, dirate = 1)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid') 

        self.rebnconv5 = REBNCONV(mid_ch, dirate = 1)

        self.rebnconv6 = REBNCONV(mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv4d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv3d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv1d = REBNCONV(out_ch, dirate = 1)

    def __call__(self,x,training):

        hx = x

        hxin = self.rebnconvin(hx,training)

        hx1 = self.rebnconv1(hxin,training)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx,training)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx,training)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx,training)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx,training)

        hx6 = self.rebnconv6(hx5,training)


        hx5d =  self.rebnconv5d(tf.concat([hx6,hx5],3),training)
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(tf.concat([hx5dup,hx4],3),training)
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(tf.concat([hx4dup,hx3],3),training)
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(tf.concat([hx3dup,hx2],3),training)
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(tf.concat([hx2dup,hx1],3),training)

        return hx1d + hxin


### RSU-5 ###
class RSU5(tf.keras.layers.Layer):#UNet05DRES(nn.Module):

    def __init__(self, mid_ch = 12, out_ch = 3, name = None):
        super(RSU5,self).__init__(name = name)

        self.rebnconvin = REBNCONV(out_ch, dirate = 1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate = 1)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid') 

        self.rebnconv2 = REBNCONV(mid_ch, dirate = 1)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.rebnconv3 = REBNCONV( mid_ch, dirate = 1)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.rebnconv4 = REBNCONV(mid_ch, dirate = 1)

        self.rebnconv5 = REBNCONV(mid_ch, dirate = 2)

        self.rebnconv4d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv3d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv1d = REBNCONV(out_ch, dirate = 1)

    def __call__(self,x,training):

        hx = x

        hxin = self.rebnconvin(hx,training)

        hx1 = self.rebnconv1(hxin,training)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx,training)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx,training)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx,training)

        hx5 = self.rebnconv5(hx4,training)

        hx4d = self.rebnconv4d(tf.concat([hx5,hx4],3),training)
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(tf.concat([hx4dup,hx3],3),training)
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(tf.concat([hx3dup,hx2],3),training)
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(tf.concat([hx2dup,hx1],3),training)

        return hx1d + hxin
    

### RSU-4 ###
class RSU4(tf.Module):#UNet04DRES(nn.Module):

    def __init__(self, mid_ch = 12, out_ch = 3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(out_ch, dirate = 1)

        self.rebnconv1 = REBNCONV(mid_ch, dirate = 1)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.rebnconv2 = REBNCONV(mid_ch, dirate = 1)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.rebnconv3 = REBNCONV(mid_ch, dirate = 1)

        self.rebnconv4 = REBNCONV(mid_ch, dirate = 2)

        self.rebnconv3d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv2d = REBNCONV(mid_ch, dirate = 1)
        self.rebnconv1d = REBNCONV(out_ch, dirate = 1)

    def __call__(self,x,training):

        hx = x

        hxin = self.rebnconvin(hx,training)

        hx1 = self.rebnconv1(hxin,training)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx,training)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx,training)

        hx4 = self.rebnconv4(hx3,training)

        hx3d = self.rebnconv3d(tf.concat([hx4,hx3],3),training)
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(tf.concat([hx3dup,hx2],3),training)
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(tf.concat([hx2dup,hx1],3),training)

        return hx1d + hxin


### RSU-4F ###
class RSU4F(tf.Module):#UNet04FRES(nn.Module):

    def __init__(self, mid_ch=12, out_ch=3, name = None):
        super(RSU4F,self).__init__(name = name)

        self.rebnconvin = REBNCONV(out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(out_ch,dirate=1)

    def __call__(self,x,training):

        hx = x

        hxin = self.rebnconvin(hx,training)

        hx1 = self.rebnconv1(hxin,training)
        hx2 = self.rebnconv2(hx1,training)
        hx3 = self.rebnconv3(hx2,training)

        hx4 = self.rebnconv4(hx3,training)

        hx3d = self.rebnconv3d(tf.concat([hx4,hx3],3),training)
        hx2d = self.rebnconv2d(tf.concat([hx3d,hx2],3),training)
        hx1d = self.rebnconv1d(tf.concat([hx2d,hx1],3),training)

        return hx1d + hxin


##### U^2-Net ####
class U2NET(tf.keras.Model):

    def __init__(self,out_ch=1, name = None):
        super(U2NET,self).__init__(name = name)

        self.stage1 = RSU7(32,64)
        self.pool12 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.stage2 = RSU6(32,128)
        self.pool23 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.stage3 = RSU5(64,256)
        self.pool34 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.stage4 = RSU4(128,512)
        self.pool45 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.stage5 = RSU4F(256,512)
        self.pool56 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.stage6 = RSU4F(256,512)

        # decoder
        self.stage5d = RSU4F(256,512)
        self.stage4d = RSU4(128,256)
        self.stage3d = RSU5(64,128)
        self.stage2d = RSU6(32,64)
        self.stage1d = RSU7(16,64)

        self.side1 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 
        self.side2 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 
        self.side3 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 
        self.side4 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 
        self.side5 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 
        self.side6 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 

        self.outconv = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 1, padding = 'same', dilation_rate = 1) 

    def call(self,x,training=True):

        hx = x

        #stage 1
        hx1 = self.stage1(hx,training)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx,training)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx,training)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx,training)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx,training)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx,training)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(tf.concat((hx6up,hx5),3),training)
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(tf.concat([hx5dup,hx4],3),training)
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(tf.concat([hx4dup,hx3],3),training)
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(tf.concat([hx3dup,hx2],3),training)
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(tf.concat([hx2dup,hx1],3),training)


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(tf.concat([d1,d2,d3,d4,d5,d6],3))

        return tf.nn.sigmoid(d0), tf.nn.sigmoid(d1), tf.nn.sigmoid(d2), tf.nn.sigmoid(d3), tf.nn.sigmoid(d4), tf.nn.sigmoid(d5), tf.nn.sigmoid(d6)


### U^2-Net small ###
class U2NETP(tf.keras.Model):

    def __init__(self,out_ch=1, name = None):
        super(U2NETP,self).__init__(name = name)

        self.stage1 = RSU7(16,64)
        self.pool12 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.stage2 = RSU6(16,64)
        self.pool23 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.stage3 = RSU5(16,64)
        self.pool34 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.stage4 = RSU4(16,64)
        self.pool45 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.stage5 = RSU4F(16,64)
        self.pool56 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid')

        self.stage6 = RSU4F(16,64)

        # decoder
        self.stage5d = RSU4F(16,64)
        self.stage4d = RSU4(16,64)
        self.stage3d = RSU5(16,64)
        self.stage2d = RSU6(16,64)
        self.stage1d = RSU7(16,64)

        self.side1 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 
        self.side2 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 
        self.side3 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 
        self.side4 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 
        self.side5 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 
        self.side6 = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 3, padding = 'same', dilation_rate = 1) 

        self.outconv = tf.keras.layers.Conv2D(filters = out_ch, kernel_size = 1, padding = 'same', dilation_rate = 1) 

    def __call__(self,x,training):

        hx = x

        #stage 1
        hx1 = self.stage1(hx,training)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx,training)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx,training)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx,training)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx,training)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx,training)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(tf.concat([hx6up,hx5],3),training)
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(tf.concat([hx5dup,hx4],3),training)
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(tf.concat([hx4dup,hx3],3),training)
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(tf.concat([hx3dup,hx2],3),training)
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(tf.concat([hx2dup,hx1],3),training)


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(tf.concat([d1,d2,d3,d4,d5,d6],3))

        return tf.nn.sigmoid(d0), tf.nn.sigmoid(d1), tf.nn.sigmoid(d2), tf.nn.sigmoid(d3), tf.nn.sigmoid(d4), tf.nn.sigmoid(d5), tf.nn.sigmoid(d6)



