import taichi as ti
import math

#python-scope
def Identity():
    return ti.Vector([0.0,0.0,0.0,1.0],dt=ti.f32)

# ================================= Operator ===================================== #
@ti.func
def Add(q1, q2): 
    return q1 + q2


@ti.func
def Sub(q1, q2):
    return q1 - q2


@ti.func
def Multiply(q1, q2):                        
    q1w, q1x, q1y, q1z = q1[3], q1[0], q1[1], q1[2]
    q2w, q2x, q2y, q2z = q2[3], q2[0], q2[1], q2[2]
    return ti.Vector([q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y,
                       q1w * q2y + q1y * q2w + q1z * q2x - q1x * q2z,
                       q1w * q2z + q1z * q2w + q1x * q2y - q1y * q2x,
                       q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z])


@ti.func
def Sacle(q, scale): 
    return q * scale


@ti.func
def Conjugate(q):  
    return ti.Vector([-q[0], -q[1], -q[2], q[3]])


@ti.func
def Inverse(q):
    return ti.Vector([-q[0], -q[1], -q[2], q[3]]).normalized()


@ti.func
def Normalized(q):
    qr = ti.Matrix.zero(float, 4, 1)
    if q.norm() > 0.:
        qr = q.normalized()
    return qr

@ti.func
def NormalizedPy(q):
    qr = ti.Matrix.zero(float, 4, 1)
    if q.norm() > 0.:
        qr = q.normalized()
    return qr

# ============================================== Method ========================================== #
@ti.func
def SetToRotate(q):
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    return ti.Matrix([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw),0.0], 
                      [2 * (qx * qy - qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qx * qw),0.0], 
                      [2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx * qx + qy * qy),0.0],
                      [0.0, 0.0, 0.0, 1.0]])


@ti.func
def Rotate(q, vec):
    q_inv = Conjugate(q)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    qwi, qxi, qyi, qzi = q_inv[3], q_inv[0], q_inv[1], q_inv[2]
    vx, vy, vz = vec[0], vec[1], vec[2]

    tx = qwi * vx + qyi * vz - qzi * vy
    ty = qwi * vy + qzi * vx - qxi * vz
    tz = qwi * vz + qxi * vy - qyi * vx
    tw = - qxi * vx - qyi * vy - qzi * vz

    return ti.Vector([tw * qx + tx * qw + ty * qz - tz * qy,
                      tw * qy + ty * qw + tz * qx - tx * qz,
                      tw * qz + tz * qw + tx * qy - ty * qx])

    
@ti.func
def GetAxis(q):
    axis = ti.Matrix.zero(float, 3, 1)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    den = 1. - qw * qw
    if den > 1e-8:
        scale = 1. / ti.sqrt(den)
        axis = ti.Vector(qx, qy, qz) * scale
    return axis

@ti.func
def GetQuatAixs0(q):
    x2=q[0] * 2.0
    w2=q[3] * 2.0

    return ti.Vector([(q[3] * w2)-1.0 + q[0] * x2,
                      (q[2] * w2) + q[1] * x2,
                      (-q[1] * w2) + q[2] * x2])  

@ti.func
def GetQuatAixs1(q):
    y2=q[1] * 2.0
    w2=q[3] * 2.0
    return ti.Vector([(-q[2] * w2) + q[0] * y2,
                      (q[3] * w2) - 1.0 + q[1] * y2,
                      (q[0] * w2) + q[2] * y2])  

@ti.func
def GetQuatAixs2(q):
    z2=q[2] * 2.0
    w2=q[3] * 2.0

    return ti.Vector([(q[1] * w2) + q[0] * z2,
                      (-q[0] * w2) + q[1] * z2,
                      (q[3] * w2) - 1.0 + q[2] * z2])  

@ti.func
def Length(q):
    return q.norm()


@ti.func
def Radians(q):
    return 2. * ti.acos(q[3])


# ======================================== Set up Quaternion ============================================ #
@ti.func
def SetDQ(q, omega):
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    ox, oy, oz = omega[0], omega[1], omega[2]
    return 0.5 * ti.Vector([ox * qw - oy * qz + oz * qy,
                            oy * qw - oz * qx + ox * qz,
                            oz * qw - ox * qy + oy * qx,
                            -ox * qx - oy * qy - oz * qz])


@ti.func
def SetFromValue(qx, qy, qz, qw):
    return ti.Vector([qx, qy, qz, qw])


@ti.func
def SetFromAxisAngle(axis, radians):
    leng = Length(axis)
    halfSin = ti.sin(0.5 * radians) / leng
    return ti.Vector([axis[0] * halfSin,
                      axis[1] * halfSin,
                      axis[2] * halfSin,
                      ti.cos(0.5 * radians)])

def SetFromAxisAnglePy(axis, radians):
    leng = axis.norm()
    halfSin = math.sin(0.5 * radians) / leng
    return ti.Vector([axis[0] * halfSin,
                      axis[1] * halfSin,
                      axis[2] * halfSin,
                      ti.cos(0.5 * radians)])


@ti.func
def SetFromEuler(yaw, pitch, roll):
    halfYaw = yaw * 0.5
    halfPitch = pitch * 0.5
    halfRoll = roll * 0.5
    cosYaw = ti.cos(halfYaw)
    sinYaw = ti.sin(halfYaw)
    cosPitch = ti.cos(halfPitch)
    sinPitch = ti.sin(halfPitch)
    cosRoll = ti.cos(halfRoll)
    sinRoll = ti.sin(halfRoll)
    return ti.Vector([cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,
                      cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,
                      sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,
                      cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw])

  

@ti.func
def SetFromTwoVec(vec1, vec2):
    v1 = vec1.normalized()
    v2 = vec2.normalized()

    c = v1.dot(v2)
    angle = ti.acos(c)
    axis = v1.cross(v2)

    if ti.abs(1.0 + c) < 1e-8:
        angle = math.pi
        if v1[0] > v1[1] and v1[0] > v1[2]:
            axis = v1.cross(ti.Vector([0., 1., 0.]))
        else:
          axis = v1.cross(ti.Vector([1., 0., 0.]))
    elif ti.abs(1.0 - c) < 1e-8:
        angle = 0.
        axis = ti.Vector([1., 0., 0.])

    return SetFromAxisAngle(axis.normalized(), angle)
