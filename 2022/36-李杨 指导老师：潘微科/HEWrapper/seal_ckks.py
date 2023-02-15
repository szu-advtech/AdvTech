import numpy as np

from seal import *

from cvxpy.cvxcore.python.cvxcore import DoubleVector


class CKKSCiphertext(Ciphertext):

    def __init__(self, public_key):
        self.public_key = public_key
        super(CKKSCiphertext, self).__init__()

    def rescale(self):
        self.public_key.evaluator.rescale_to_next_inplace(self)

    def lazy_rescale(self, x1, x2, mode='mul'):
        if x1.scale() == x2.scale():
            if mode == 'mul' and x1.scale() > self.public_key.scale:
                self.public_key.evaluator.rescale_to_next_inplace(x1)
                self.public_key.evaluator.rescale_to_next_inplace(x2)
        else:
            x1_scale = np.log2(x1.scale())
            x2_scale = np.log2(x2.scale())
            if np.abs(x1_scale - x2_scale) > 1:
                if x1_scale > x2_scale:
                    self.public_key.evaluator.rescale_to_next_inplace(x1)
                else:
                    self.public_key.evaluator.rescale_to_next_inplace(x2)
                x1_scale = np.log2(x1.scale())
                x2_scale = np.log2(x2.scale())
            if x1_scale != x2_scale and np.abs(x1_scale - x2_scale) < 1:
                if not isinstance(x1, Plaintext) and x1_scale != np.log2(self.public_key.scale):
                    x1.scale(self.public_key.scale)
                if not isinstance(x2, Plaintext) and x2_scale != np.log2(self.public_key.scale):
                    x2.scale(self.public_key.scale)
        x1_level = self.public_key.context.get_context_data(x1.parms_id()).chain_index()
        x2_level = self.public_key.context.get_context_data(x2.parms_id()).chain_index()
        if x1_level > x2_level:
            self.public_key.evaluator.mod_switch_to_inplace(x1, x2.parms_id())
        elif x1_level < x2_level:
            self.public_key.evaluator.mod_switch_to_inplace(x2, x1.parms_id())
        else:
            pass
        return x1, x2

    def __mul__(self, x):
        if isinstance(x, CKKSCiphertext):
            c_times_c = True
        else:
            if not isinstance(x, Plaintext):
                x = self.public_key.encode(x)
            c_times_c = False
        if self.public_key.lazy_rescale:
            x1, x2 = self.lazy_rescale(self, x, mode='mul')
        else:
            x1, x2 = self, x
        result = CKKSCiphertext(self.public_key)
        if c_times_c:
            self.public_key.evaluator.multiply(x1, x2, result)
            self.public_key.evaluator.relinearize_inplace(result, self.public_key.relin_keys)
        else:
            self.public_key.evaluator.multiply_plain(x1, x2, result)
        if not self.public_key.lazy_rescale:
            self.public_key.evaluator.rescale_to_next_inplace(result)
        return result

    def __add__(self, x):
        if isinstance(x, CKKSCiphertext):
            c_add_c = True
        else:
            if not isinstance(x, Plaintext):
                x = self.public_key.encode(x)
            c_add_c = False
        # Lazy rescale
        if self.public_key.lazy_rescale:
            x1, x2 = self.lazy_rescale(self, x, mode='add')
        else:
            x1, x2 = self, x
        result = CKKSCiphertext(self.public_key)
        if c_add_c:
            self.public_key.evaluator.add(x1, x2, result)
        else:
            self.public_key.evaluator.add_plain(x1, x2, result)
        return result

    def __iadd__(self, x):
        if isinstance(x, CKKSCiphertext):
            c_add_c = True
        else:
            if not isinstance(x, Plaintext):
                x = self.public_key.encode(x)
            c_add_c = False
        # Lazy rescale
        if self.public_key.lazy_rescale:
            x1, x2 = self.lazy_rescale(self, x, mode='add')
        else:
            x1, x2 = self, x
        if c_add_c:
            self.public_key.evaluator.add_inplace(x1, x2)
        else:
            self.public_key.evaluator.add_plain_inplace(x1, x2)
        return x1

    def __sub__(self, x):
        return self + (-1 * x)


class CKKSPublicKey:
    def __init__(self, context, public_key, relin_keys, scale, params, lazy_rescale):
        self.context = context
        self.encryptor = Encryptor(context, public_key)
        self.evaluator = Evaluator(context)
        self.encoder = CKKSEncoder(context)
        self.slot_count = self.encoder.slot_count()
        self.scale = scale
        self.relin_keys = relin_keys
        self.params = params
        self.lazy_rescale = lazy_rescale

    def encrypt(self, raw):
        encoded_plaintext = self.encode(raw)
        encrypted_ciphertext = self.encryptor.encrypt(encoded_plaintext)
        return encrypted_ciphertext

    def encode(self, raw):
        plain_row = self.encoder.encode(raw, self.scale)
        return plain_row;

    def decode(self, plaintext):
        result = DoubleVector()
        self.encoder.decode(plaintext, result)
        return np.array(result)

    def format_params(self):
        return '%s-%s-[%s]' % (
            self.params.poly_modulus_degree(), str(int(np.round(np.log2(self.scale)))),
            ','.join([str(int(np.round(np.log2(e.value())))) for e in self.params.coeff_modulus()])
        )


class CKKSSecretKey:
    def __init__(self, context, secret_key, params):
        self.context = context
        self.encoder = CKKSEncoder(self.context)
        self.decryptor = Decryptor(context, secret_key)
        self.params = params

    def decrypt(self, ciphertext):
        encoded_plaintext = self.decryptor.decrypt(ciphertext)
        result = self.encoder.decode(encoded_plaintext)
        return np.array(result)


def generate_pk_and_sk(poly_modulus_degree, scale=None, coefficient_modulus=None, lazy_rescale=True):

    if scale is None or coefficient_modulus is None:
        if poly_modulus_degree == 4096:
            scale = 2**24
            coefficient_modulus = [30, 24, 24, 30]
        elif poly_modulus_degree == 8192:
            scale = 2**40
            coefficient_modulus = [60, 40, 40, 60]
        elif poly_modulus_degree == 16384:
            scale = 2**50
            coefficient_modulus = [60, 50, 50, 50, 50, 50, 50, 60]
        elif poly_modulus_degree == 32768:
            scale = 2 ** 60
            coefficient_modulus = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
        else:
            raise ValueError('Please provide scale and coefficient_modulus')

    parms = EncryptionParameters(scheme_type.ckks)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, coefficient_modulus))
    context = SEALContext(parms)
    keygen = KeyGenerator(context)

    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    relin_keys = keygen.create_relin_keys()
    evaluator = Evaluator(context)
    return CKKSPublicKey(context, public_key, relin_keys, scale, parms, lazy_rescale),\
           CKKSSecretKey(context, secret_key, parms),\
           evaluator

