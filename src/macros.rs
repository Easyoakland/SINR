//! Various macros. The `u` prefixed macros such as `uassert!` avoid performing any checks if the `unsafe` feature is active.

#![allow(unused_macros)]

macro_rules! trace {
    (file $fmt:expr, $($name:expr),* ; $val:expr $(,)?) => {
        #[cfg(feature = "trace")]
        {
            std::fs::write(format!($fmt, $($name),*), $val).unwrap();
        }
    };
    ($fmt:expr $(,)? $(, $($name:expr),+ $(,)?)?) => {
        #[cfg(feature = "trace")]
        {
            let f = format!($fmt, $($($name),+)?);
            eprintln!("[{}:{}:{}] {f}", file!(), line!(), column!())
        }
    };
}
pub(super) use trace;

/// Unsafe assert.
macro_rules! uassert {
    ($e:expr) => {{
        #[cfg(all(not(feature = "unsafe"), debug_assertions))]
        {
            assert!($e)
        }
        #[cfg(feature = "unsafe")]
        {
            unsafe { core::hint::assert_unchecked($e) }
        }
    }};
}
pub(super) use uassert;

/// Unsafe unreachable
macro_rules! uunreachable {
    () => {{
        #[cfg(not(feature = "unsafe"))]
        let _res = unreachable!();
        #[cfg(feature = "unsafe")]
        let _res = unsafe { core::hint::unreachable_unchecked() };
        #[allow(unreachable_code)]
        _res
    }};
}
pub(super) use uunreachable;
