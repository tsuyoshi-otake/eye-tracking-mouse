//! Drawing a window's content with GDI.
//!
//! Contract: the caller describes everything the window shows as a list of
//! shapes, in the order they are painted (later ones cover earlier ones).
//! `Surface::paint` draws them off-screen and shows the result in one step,
//! so nothing half-drawn is ever seen. Positions are pixels of the window's
//! inside, counted from its top left corner.

use std::mem::size_of;

use windows::Win32::Foundation::{COLORREF, RECT};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CLEARTYPE_QUALITY, CLIP_DEFAULT_PRECIS, CreateCompatibleBitmap,
    CreateCompatibleDC, CreateFontW, DC_BRUSH, DC_PEN, DEFAULT_CHARSET, DIB_RGB_COLORS, DT_CENTER, DT_NOCLIP,
    DT_NOPREFIX, DT_SINGLELINE, DT_VCENTER, DeleteDC, DeleteObject, DrawTextW, Ellipse, FW_SEMIBOLD, FillRect, GetDC,
    GetStockObject, HALFTONE, HBITMAP, HBRUSH, HDC, HGDIOBJ, OUT_DEFAULT_PRECIS, ReleaseDC, SRCCOPY, SelectObject,
    SetBkMode, SetDCBrushColor, SetDCPenColor, SetStretchBltMode, SetTextColor, StretchDIBits, TRANSPARENT,
};
use windows::core::w;

use crate::geometry::RectI;

use super::window::{self, Window};

/// Red, green, blue.
pub type Colour = (u8, u8, u8);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape<'a> {
    /// A filled rectangle.
    Box { area: RectI, colour: Colour },
    /// A filled circle.
    Disc { centre: (i32, i32), radius: i32, colour: Colour },
    /// One line of text in the middle of `area`, its letters `height` pixels tall.
    Text { text: &'a str, area: RectI, height: i32, colour: Colour },
    /// A picture stretched over `area`: `size.0 * size.1` pixels, row by row from the top, each 0x00RRGGBB.
    Picture { pixels: &'a [u32], size: (usize, usize), area: RectI },
}

fn colour_ref(colour: Colour) -> COLORREF {
    COLORREF(u32::from(colour.0) | u32::from(colour.1) << 8 | u32::from(colour.2) << 16)
}

fn rect(area: RectI) -> RECT {
    RECT { left: area.left, top: area.top, right: area.right, bottom: area.bottom }
}

/// The off-screen picture and the device context that draws on it.
struct Buffer {
    memory: HDC,
    bitmap: HBITMAP,
    /// What the context held before `bitmap`; it is put back before the context goes.
    original: HGDIOBJ,
    size: (i32, i32),
}

impl Buffer {
    /// SAFETY: `like` must be a live device context.
    unsafe fn new(like: HDC, size: (i32, i32)) -> Option<Buffer> {
        unsafe {
            let memory = CreateCompatibleDC(Some(like));
            if memory.is_invalid() {
                return None;
            }
            let bitmap = CreateCompatibleBitmap(like, size.0, size.1);
            if bitmap.is_invalid() {
                let _ = DeleteDC(memory);
                return None;
            }
            let original = SelectObject(memory, bitmap.into());
            // These two take their colour from the context, so no shape needs a brush or pen of its own.
            SelectObject(memory, GetStockObject(DC_BRUSH));
            SelectObject(memory, GetStockObject(DC_PEN));
            SetBkMode(memory, TRANSPARENT);
            SetStretchBltMode(memory, HALFTONE);
            Some(Buffer { memory, bitmap, original, size })
        }
    }

    /// SAFETY: none beyond `self` being a buffer made by `new`.
    unsafe fn draw(&self, shape: &Shape) {
        let dc = self.memory;
        unsafe {
            match *shape {
                Shape::Box { area, colour } => {
                    SetDCBrushColor(dc, colour_ref(colour));
                    FillRect(dc, &rect(area), HBRUSH(GetStockObject(DC_BRUSH).0));
                }
                Shape::Disc { centre, radius, colour } => {
                    SetDCBrushColor(dc, colour_ref(colour));
                    SetDCPenColor(dc, colour_ref(colour));
                    // The right and bottom edges are not part of the ellipse.
                    let _ = Ellipse(dc, centre.0 - radius, centre.1 - radius, centre.0 + radius + 1, centre.1 + radius + 1);
                }
                Shape::Text { text, area, height, colour } => {
                    let font = CreateFontW(
                        -height,
                        0,
                        0,
                        0,
                        FW_SEMIBOLD.0 as i32,
                        0,
                        0,
                        0,
                        DEFAULT_CHARSET,
                        OUT_DEFAULT_PRECIS,
                        CLIP_DEFAULT_PRECIS,
                        CLEARTYPE_QUALITY,
                        0,
                        w!("Yu Gothic UI"),
                    );
                    // Without the font the text is still drawn, in whatever the context has.
                    let previous = (!font.is_invalid()).then(|| SelectObject(dc, font.into()));
                    SetTextColor(dc, colour_ref(colour));
                    let mut letters: Vec<u16> = text.encode_utf16().collect();
                    let format = DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX | DT_NOCLIP;
                    DrawTextW(dc, &mut letters, &mut rect(area), format);
                    if let Some(previous) = previous {
                        SelectObject(dc, previous);
                        let _ = DeleteObject(font.into());
                    }
                }
                Shape::Picture { pixels, size, area } => {
                    if size.0 == 0 || size.1 == 0 || pixels.len() != size.0 * size.1 {
                        return;
                    }
                    let header = BITMAPINFOHEADER {
                        biSize: size_of::<BITMAPINFOHEADER>() as u32,
                        biWidth: size.0 as i32,
                        biHeight: -(size.1 as i32), // top-down
                        biPlanes: 1,
                        biBitCount: 32,
                        biCompression: BI_RGB.0,
                        ..Default::default()
                    };
                    let info = BITMAPINFO { bmiHeader: header, ..Default::default() };
                    // SAFETY: `pixels` holds exactly the `size.0 * size.1` 32-bit pixels the header promises.
                    StretchDIBits(
                        dc,
                        area.left,
                        area.top,
                        area.width(),
                        area.height(),
                        0,
                        0,
                        size.0 as i32,
                        size.1 as i32,
                        Some(pixels.as_ptr().cast()),
                        &info,
                        DIB_RGB_COLORS,
                        SRCCOPY,
                    );
                }
            }
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            SelectObject(self.memory, self.original);
            let _ = DeleteObject(self.bitmap.into());
            let _ = DeleteDC(self.memory);
        }
    }
}

/// What a window shows. Keeps the off-screen picture between paints, as large as the window's inside.
pub struct Surface {
    window: Window,
    buffer: Option<Buffer>,
}

impl Surface {
    pub fn new(window: Window) -> Surface {
        Surface { window, buffer: None }
    }

    /// Draws `shapes` over `background` and shows the result. Does nothing while the window has no size.
    pub fn paint(&mut self, background: Colour, shapes: &[Shape]) {
        let size = window::client_size(self.window);
        if size.0 <= 0 || size.1 <= 0 {
            return;
        }
        unsafe {
            let target = GetDC(Some(self.window));
            if target.is_invalid() {
                return;
            }
            if self.buffer.as_ref().is_none_or(|buffer| buffer.size != size) {
                // The old picture goes before the new one is made, so the two never exist together.
                self.buffer = None;
                self.buffer = Buffer::new(target, size);
            }
            if let Some(buffer) = &self.buffer {
                buffer.draw(&Shape::Box { area: RectI::new(0, 0, size.0, size.1), colour: background });
                for shape in shapes {
                    buffer.draw(shape);
                }
                let _ = BitBlt(target, 0, 0, size.0, size.1, Some(buffer.memory), 0, 0, SRCCOPY);
            }
            ReleaseDC(Some(self.window), target);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::window::Kind;
    use windows::Win32::Graphics::Gdi::GetPixel;

    const BACKGROUND: Colour = (10, 20, 30);

    #[test]
    fn shapes_land_where_they_are_asked_to_in_the_colours_given() {
        window::register_reporting_class("EtmCanvasTest").unwrap();
        // Never shown: painting needs a window, not a visible one.
        let frame = window::open("EtmCanvasTest", "canvas test", Kind::Framed, RectI::new(0, 0, 480, 360)).unwrap();
        let mut surface = Surface::new(frame);
        let picture = vec![0x00C8_6432u32; 8 * 4];
        let wide = RectI::new(0, 150, 300, 200);
        surface.paint(
            BACKGROUND,
            &[
                Shape::Box { area: RectI::new(20, 30, 60, 50), colour: (255, 0, 0) },
                Shape::Disc { centre: (120, 60), radius: 10, colour: (0, 255, 0) },
                Shape::Picture { pixels: &picture, size: (8, 4), area: RectI::new(200, 20, 280, 60) },
                Shape::Text { text: "視線 Aa", area: wide, height: 30, colour: (255, 255, 255) },
                // A picture whose size does not match its pixels is left out, not read past its end.
                Shape::Picture { pixels: &picture, size: (80, 40), area: RectI::new(0, 0, 10, 10) },
            ],
        );
        let buffer = surface.buffer.as_ref().expect("the window has a size, so there is a picture of it");
        assert_eq!(buffer.size, window::client_size(frame));
        let at = |x: i32, y: i32| {
            let pixel = unsafe { GetPixel(buffer.memory, x, y) }.0;
            ((pixel & 0xFF) as u8, (pixel >> 8 & 0xFF) as u8, (pixel >> 16 & 0xFF) as u8)
        };
        assert_eq!(at(5, 5), BACKGROUND);
        assert_eq!(at(20, 30), (255, 0, 0), "the box starts at its left and top");
        assert_eq!(at(59, 49), (255, 0, 0));
        assert_eq!(at(60, 50), BACKGROUND, "and ends before its right and bottom");
        assert_eq!(at(120, 60), (0, 255, 0));
        assert_eq!(at(120, 70), (0, 255, 0), "the disc reaches its radius");
        assert_eq!(at(120, 72), BACKGROUND);
        assert_eq!(at(240, 40), (0xC8, 0x64, 0x32), "a picture's pixels are 0x00RRGGBB");
        let inked = (0..300).flat_map(|x| (150..200).map(move |y| (x, y))).filter(|&(x, y)| at(x, y) != BACKGROUND).count();
        assert!(inked > 200, "only {inked} pixels of text were drawn");

        // Painting again starts from the background: nothing of the last picture stays.
        surface.paint(BACKGROUND, &[]);
        let buffer = surface.buffer.as_ref().unwrap();
        let pixel = unsafe { GetPixel(buffer.memory, 30, 40) }.0;
        assert_eq!(pixel, colour_ref(BACKGROUND).0);
        drop(surface);
        window::destroy(frame);
    }
}
