macro_rules! def_file_id_type {
    ($name:ident ( $($field:ty),+ )) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name($($field,)+);
    };
}

#[cfg(target_family = "unix")]
mod unix {
    def_file_id_type!(UnixFileId(u64));

    pub fn get_file_id(metadata: &dyn std::os::unix::fs::MetadataExt) -> UnixFileId {
        UnixFileId(metadata.ino())
    }
}

#[cfg(target_family = "unix")]
use unix::get_file_id as get_physical_file_id;
#[cfg(target_family = "unix")]
use unix::UnixFileId as PhysicalFileId;

#[cfg(target_family = "windows")]
mod windows {
    def_file_id_type!(WindowsFileId(u64, u32));

    pub fn get_file_id(metadata: &dyn std::os::windows::fs::MetadataExt) -> WindowsFileId {
        let volume_id = metadata.volume_serial_number().unwrap();
        let file_index = metadata.file_index().unwrap();
        WindowsFileId(file_index, volume_id)
    }
}

#[cfg(target_family = "windows")]
use windows::get_file_id as get_physical_file_id;
#[cfg(target_family = "windows")]
use windows::WindowsFileId as PhysicalFileId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FileIdInner {
    None,
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
    pub fn register_file<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<FileId> {
        let metadata = std::fs::metadata(path.as_ref())?;
        assert!(metadata.is_file(), "path does not point to a file");

        let id: FileId = FileId::new_physical(get_physical_file_id(&metadata));
        if !self.files.contains_key(&id) {
            let path = path.as_ref().canonicalize()?.into_boxed_path();
            let text = std::fs::read_to_string(&path)?;

            self.files.insert(id, SourceFile::new(path, text.into()));
        }

        Ok(id)
    }

    /// Registers an in-memory file with the server and returns its ID.
    /// The file being registered is always considered to be a distinct new file regardless of its path.
    pub fn register_file_memory<P, S>(&mut self, path: P, text: S) -> FileId
    where
        P: AsRef<Path>,
        S: Into<Cow<'static, str>>,
    {
        let id: FileId = FileId::new_memory(self.next_memory_id);
        self.next_memory_id += 1;

        let path = path.as_ref().to_owned().into_boxed_path();
        self.files.insert(id, SourceFile::new(path, text.into()));

        id
    }
}
