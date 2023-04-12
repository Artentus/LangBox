macro_rules! def_file_id_type {
    ($name:ident ( $($field:ty),+ )) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name($($field,)+);
    };
}

#[cfg(target_family = "unix")]
mod unix {
    def_file_id_type!(UnixFileId(u64));

    pub fn get_file_id(metadata: &dyn std::os::unix::fs::MetadataExt) -> Option<UnixFileId> {
        Some(UnixFileId(metadata.ino()))
    }
}

#[cfg(target_family = "unix")]
use unix::get_file_id as get_physical_file_id;
#[cfg(target_family = "unix")]
use unix::UnixFileId as PhysicalFileId;

#[cfg(target_family = "windows")]
mod windows {
    def_file_id_type!(WindowsFileId(u32, u64));

    pub fn get_file_id(metadata: &dyn std::os::windows::fs::MetadataExt) -> Option<WindowsFileId> {
        let volume_id = metadata.volume_serial_number()?;
        let file_index = metadata.file_index()?;
        Some(WindowsFileId(volume_id, file_index))
    }
}

#[cfg(target_family = "windows")]
use windows::get_file_id as get_physical_file_id;
#[cfg(target_family = "windows")]
use windows::WindowsFileId as PhysicalFileId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FileIdInner {
    None,
    #[cfg(any(target_family = "unix", target_family = "windows"))]
    Physical(PhysicalFileId),
    Memory(u64),
}

/// Uniquely identifies a file on the host computer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FileId(FileIdInner);

impl FileId {
    /// Indicates no association to any file
    pub const NONE: Self = Self(FileIdInner::None);

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[inline]
    fn new_physical(id: PhysicalFileId) -> Self {
        Self(FileIdInner::Physical(id))
    }

    #[inline]
    fn new_memory(id: u64) -> Self {
        Self(FileIdInner::Memory(id))
    }
}

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

/// A file containing source code
#[derive(Debug)]
pub struct SourceFile {
    path: Box<Path>,
    text: Cow<'static, str>,
}

impl SourceFile {
    fn new(path: Box<Path>, text: Cow<'static, str>) -> Self {
        assert!(
            text.len() <= (u32::MAX as usize),
            "file `{}` is too big",
            path.display()
        );

        Self { path, text }
    }

    /// The cannonical path of the file
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The text content of the file
    #[inline]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The length of the text content
    #[inline]
    pub fn text_len(&self) -> u32 {
        self.text.len() as u32
    }
}

/// Platform independant wrapper for reading unique files
pub struct FileServer {
    files: HashMap<FileId, SourceFile>,
    memory_id_map: HashMap<Box<Path>, u64>,
    next_memory_id: u64,
}

impl FileServer {
    /// Creates a new file server.
    pub fn new() -> Self {
        let mut files = HashMap::new();

        let none_path: &Path = "<none>".as_ref();
        files.insert(
            FileId::NONE,
            SourceFile {
                path: none_path.to_owned().into_boxed_path(),
                text: "".into(),
            },
        );

        Self {
            files,
            memory_id_map: HashMap::new(),
            next_memory_id: 0,
        }
    }

    /// The files registered by the server
    #[inline]
    pub fn files(&self) -> impl Iterator<Item = &SourceFile> {
        self.files.values()
    }

    /// Gets a file that has previously been registered with the server.
    #[inline]
    pub fn get_file(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(&id)
    }

    /// Registers a file with the server and returns its ID.
    /// If the file has previously been registered no actual file system access will occur.
    ///
    /// If an in-memory file with the given path has been registered previously, it will be returned over any physical file.
    pub fn register_file<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<FileId> {
        let path = path.as_ref();
        if let Some(&id) = self.memory_id_map.get(path) {
            return Ok(FileId::new_memory(id));
        }

        #[cfg(any(target_family = "unix", target_family = "windows"))]
        {
            let metadata = std::fs::metadata(path)?;
            let Some(physical_id) = get_physical_file_id(&metadata) else {
                return Err(std::io::Error::new(std::io::ErrorKind::Other, "path does not point to a file"));
            };

            let id: FileId = FileId::new_physical(physical_id);
            if !self.files.contains_key(&id) {
                let path = path.canonicalize()?.into_boxed_path();
                let text = std::fs::read_to_string(&path)?;

                self.files.insert(id, SourceFile::new(path, text.into()));
            }

            Ok(id)
        }

        #[cfg(not(any(target_family = "unix", target_family = "windows")))]
        Err(std::io::ErrorKind::Unsupported.into())
    }

    /// Registers an in-memory file with the server and returns its ID.
    /// Returns an error if an in-memory file with the same path has already been registered.
    pub fn register_file_memory<P, S>(&mut self, path: P, text: S) -> Result<FileId, ()>
    where
        P: AsRef<Path>,
        S: Into<Cow<'static, str>>,
    {
        let path = path.as_ref();
        if self.memory_id_map.contains_key(path) {
            return Err(());
        }

        let memory_id = self.next_memory_id;
        self.next_memory_id += 1;

        let id: FileId = FileId::new_memory(memory_id);
        self.files
            .insert(id, SourceFile::new(path.into(), text.into()));
        self.memory_id_map.insert(path.into(), memory_id);

        Ok(id)
    }
}
