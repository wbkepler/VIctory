//! 64-bit floating-point (`f64`) lowering via DXIL `double`,
//! `dx.op.makeDouble.f64`, and `dx.op.splitDouble.f64`.

use super::{Lower, Why};
use crate::emit::bitcode::{Attr, Bin, Cast, V};
use crate::emit::shader::flags;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Reg};

impl<'a> Lower<'a> {
    /// Converts or computes a GFX10 64-bit float (`f64`) VALU instruction.
    /// Returns `Some(Ok(()))` if handled, `None` for non-f64 instructions.
    pub(super) fn valu_f64(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
    ) -> Option<Result<(), Why>> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let c = self.cur;
        match name {
            "v_cvt_f32_f64" => Some((|| {
                let a = self.f64_operand(inst, 0)?;
                let r = self.s.m.cast(c, Cast::FpTrunc, a, self.s.t.f32);
                self.write_f(inst, r, guard)
            })()),
            "v_cvt_f64_f32" => Some((|| {
                self.s.flags |= flags::DOUBLES;
                let a = self.f32_src(inst, 0)?;
                let f64t = self.s.m.double();
                let r = self.s.m.cast(c, Cast::FpExt, a, f64t);
                self.write_f64(inst, v, r, guard)
            })()),
            "v_cvt_f64_i32" | "v_cvt_f64_u32" => Some((|| {
                self.s.flags |= flags::DOUBLES;
                let op_in = *v.src.first().ok_or("a missing operand")?;
                let a = self.operand(inst, &op_in)?;
                let op = if name == "v_cvt_f64_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let f64t = self.s.m.double();
                let r = self.s.m.cast(c, op, a, f64t);
                self.write_f64(inst, v, r, guard)
            })()),
            "v_add_f64" | "v_mul_f64" => Some((|| {
                let a = self.f64_operand(inst, 0)?;
                let bb = self.f64_operand(inst, 1)?;
                let op = if name == "v_add_f64" {
                    Bin::Add
                } else {
                    Bin::Mul
                };
                let r = self.s.m.bin(self.cur, op, a, bb);
                self.write_f64(inst, v, r, guard)
            })()),
            "v_fma_f64" => Some((|| {
                let a = self.f64_operand(inst, 0)?;
                let bb = self.f64_operand(inst, 1)?;
                let dd = self.f64_operand(inst, 2)?;
                let f64t = self.s.m.double();
                let r = self
                    .s
                    .op(self.cur, "tertiary.f64", Attr::ReadNone, f64t, 47, &[a, bb, dd]);
                self.write_f64(inst, v, r, guard)
            })()),
            _ => None,
        }
    }

    /// A 32-bit float source operand `k` with `abs` / `neg` applied.
    fn f32_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = v.src.get(k).ok_or("a missing operand")?;
        let mut x = self.operand(inst, op)?;
        let c = self.cur;
        if v.abs >> k & 1 != 0 {
            let mask = self.s.m.i32(0x7FFF_FFFF);
            x = self.s.m.bin(c, Bin::And, x, mask);
        }
        if v.neg >> k & 1 != 0 {
            let sign = self.s.m.i32(0x8000_0000);
            x = self.s.m.bin(c, Bin::Xor, x, sign);
        }
        Ok(self.as_float(x))
    }

    /// A 64-bit float (`f64`) source operand `k` as DXIL `double`, with its
    /// `abs` / `neg` modifiers applied to the high dword.
    fn f64_operand(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit float SDWA or DPP modifiers");
        }
        self.s.flags |= flags::DOUBLES;
        let c = self.cur;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<(V, V), Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            Ok((lo, hi))
        };
        let (lo, mut hi) = match op {
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))?
            }
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1))?,
            Operand::Const(n) => {
                let mut bits = match n {
                    128..=192 => f64::from(n - 128).to_bits(),
                    193..=208 => (-f64::from(n - 192)).to_bits(),
                    240..=248 => [
                        0.5f64,
                        -0.5,
                        1.0,
                        -1.0,
                        2.0,
                        -2.0,
                        4.0,
                        -4.0,
                        0.159_154_943_091_895_35,
                    ][usize::from(n - 240)]
                        .to_bits(),
                    _ => return Err("a special 64-bit float operand"),
                };
                if v.abs >> k & 1 != 0 {
                    bits &= !(1u64 << 63);
                }
                if v.neg >> k & 1 != 0 {
                    bits ^= 1u64 << 63;
                }
                return Ok(self.s.m.f64(bits));
            }
            // GFX10 32-bit literals in 64-bit float ops fill bits 32-63.
            Operand::Literal(x) => {
                let mut bits = u64::from(x) << 32;
                if v.abs >> k & 1 != 0 {
                    bits &= !(1u64 << 63);
                }
                if v.neg >> k & 1 != 0 {
                    bits ^= 1u64 << 63;
                }
                return Ok(self.s.m.f64(bits));
            }
            _ => return Err("a 64-bit float operand"),
        };
        if v.abs >> k & 1 != 0 {
            let mask = self.s.m.i32(0x7FFF_FFFF);
            hi = self.s.m.bin(c, Bin::And, hi, mask);
        }
        if v.neg >> k & 1 != 0 {
            let sign = self.s.m.i32(0x8000_0000);
            hi = self.s.m.bin(c, Bin::Xor, hi, sign);
        }
        let f64t = self.s.m.double();
        self.s.flags |= flags::DOUBLE_EXTENSIONS;
        Ok(self
            .s
            .op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[lo, hi]))
    }

    /// Writes a 64-bit float (`f64`) result `r` to a VGPR pair (`v.dst`),
    /// honoring `v.clamp` (`saturate`).
    fn write_f64(
        &mut self,
        inst: &Inst,
        v: &crate::gfx10::Valu,
        mut r: V,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 64-bit result outside a VGPR");
        };
        self.s.flags |= flags::DOUBLE_EXTENSIONS;
        let c = self.cur;
        let f64t = self.s.m.double();
        if v.clamp {
            r = self
                .s
                .op(c, "unary.f64", Attr::ReadNone, f64t, 7, &[r]);
        }
        let i32t = self.s.t.i32;
        let pair_ty = self.s.m.named_struct("dx.types.splitdouble", vec![i32t, i32t]);
        let parts = self
            .s
            .op(c, "splitDouble.f64", Attr::ReadNone, pair_ty, 102, &[r]);
        let lo = self.s.m.extract(c, parts, 0);
        let hi = self.s.m.extract(c, parts, 1);
        self.write_placed(inst, Reg::V(reg), lo, guard);
        self.write_placed(inst, Reg::V(reg + 1), hi, guard);
        Ok(())
    }
}
