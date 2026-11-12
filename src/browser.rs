//! The app's Open dialog: browse folders for .wgsl shapes and .vti grids, or create a new shape
//! from the template. Plain egui (no native dialog), so it works the same on every desktop.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub struct Browser {
    pub open: bool,
    pub dir: PathBuf,
    new_name: String,
    error: Option<String>,
    entries: Vec<Entry>,
    /// the folder `entries` was read from
    listed: Option<PathBuf>,
}

struct Entry {
    name: String,
    path: PathBuf,
    dir: bool,
    modified: Option<SystemTime>,
}

/// What the user picked.
pub enum Pick {
    Open(PathBuf),
    /// a new shape was created; open it and start the editor
    Created(PathBuf),
}

pub fn openable(p: &Path) -> bool {
    p.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("wgsl") || e.eq_ignore_ascii_case("vti"))
}

pub fn ago(t: Option<SystemTime>) -> String {
    let Some(s) = t.and_then(|t| t.elapsed().ok()).map(|d| d.as_secs()) else {
        return String::new();
    };
    match s {
        0..60 => "just now".into(),
        60..3600 => format!("{} min ago", s / 60),
        3600..86400 => format!("{} h ago", s / 3600),
        _ => format!("{} d ago", s / 86400),
    }
}

impl Browser {
    pub fn new(dir: PathBuf) -> Browser {
        Browser {
            open: false,
            dir,
            new_name: String::new(),
            error: None,
            entries: vec![],
            listed: None,
        }
    }

    pub fn show_at(&mut self, dir: &Path) {
        self.dir = dir.to_path_buf();
        self.open = true;
        self.listed = None;
        self.error = None;
    }

    fn refresh(&mut self) {
        self.listed = Some(self.dir.clone());
        self.entries.clear();
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            self.error = Some(format!("cannot read {}", self.dir.display()));
            return;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || name == "target" {
                continue;
            }
            let dir = path.is_dir();
            if !dir && !openable(&path) {
                continue;
            }
            let modified = e.metadata().and_then(|m| m.modified()).ok();
            self.entries.push(Entry {
                name,
                path,
                dir,
                modified,
            });
        }
        // folders first, then files, newest first: what you worked on last is on top
        self.entries.sort_by(|a, b| {
            b.dir.cmp(&a.dir).then_with(|| {
                if a.dir {
                    a.name.cmp(&b.name)
                } else {
                    b.modified.cmp(&a.modified)
                }
            })
        });
    }

    /// Draw the dialog if it's open. `can_close` is false while nothing is loaded yet.
    pub fn ui(&mut self, ctx: &egui::Context, can_close: bool) -> Option<Pick> {
        if !self.open {
            return None;
        }
        if self.listed.as_ref() != Some(&self.dir) {
            self.refresh();
        }
        let mut pick = None;
        let mut go: Option<PathBuf> = None;
        let modal = egui::Modal::new(egui::Id::unique("open_dialog")).show(ctx, |ui| {
            ui.set_width(520.0);
            ui.heading("Open a shape");
            ui.weak(".wgsl shapes and .vti grids (e.g. a simulation result)");
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Up").clicked()
                    && let Some(p) = self.dir.parent()
                {
                    go = Some(p.to_path_buf());
                }
                let cwd = std::env::current_dir().unwrap_or_default();
                for (label, d) in [
                    ("project", cwd.clone()),
                    ("shapes", cwd.join("shapes")),
                    ("out", cwd.join("out")),
                ] {
                    if d.is_dir() && ui.selectable_label(self.dir == d, label).clicked() {
                        go = Some(d);
                    }
                }
            });
            ui.monospace(self.dir.display().to_string());
            ui.separator();
            egui::ScrollArea::vertical()
                .max_height(360.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    if self.entries.is_empty() {
                        ui.weak("no folders, .wgsl or .vti files here");
                    }
                    egui::Grid::new("entries").num_columns(2).striped(true).show(ui, |ui| {
                        for e in &self.entries {
                            let label = if e.dir { format!("{}/", e.name) } else { e.name.clone() };
                            let text = if e.dir {
                                egui::RichText::new(label)
                            } else {
                                egui::RichText::new(label).strong()
                            };
                            if ui.add(egui::Button::new(text).frame(false)).clicked() {
                                if e.dir {
                                    go = Some(e.path.clone());
                                } else {
                                    pick = Some(Pick::Open(e.path.clone()));
                                }
                            }
                            ui.weak(if e.dir { String::new() } else { ago(e.modified) });
                            ui.end_row();
                        }
                    });
                });
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("New shape");
                let r = ui.add(
                    egui::TextEdit::singleline(&mut self.new_name)
                        .hint_text("name")
                        .desired_width(160.0),
                );
                let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if ui
                    .add_enabled(!self.new_name.trim().is_empty(), egui::Button::new("Create"))
                    .on_hover_text(
                        "an empty .wgsl in this folder: build it from primitives in Design, or click \
                         Edit to script it",
                    )
                    .clicked()
                    || (enter && !self.new_name.trim().is_empty())
                {
                    match crate::shape::create_empty(&self.dir, &self.new_name) {
                        Ok(p) => {
                            self.new_name.clear();
                            pick = Some(Pick::Created(p));
                        }
                        Err(e) => self.error = Some(format!("{e:#}")),
                    }
                }
                if can_close {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Cancel").clicked() {
                            self.open = false;
                        }
                    });
                }
            });
            if let Some(e) = &self.error {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
        });
        if can_close && modal.should_close() {
            self.open = false;
        }
        if let Some(d) = go {
            self.dir = d;
            self.error = None;
        }
        if pick.is_some() {
            self.open = false;
            self.error = None;
        }
        pick
    }
}
