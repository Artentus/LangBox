use crate::SharedStr;
use std::borrow::Cow;
use std::path::Path;

#[cfg(target_family = "unix")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct PhysicalFileId(u64);

#[cfg(target_family = "unix")]
#[inline]
fn get_physical_file_id(metadata: &dyn std::os::unix::fs::MetadataExt) -> Option<PhysicalFileId> {
    Some(PhysicalFileId(metadata.ino()))
}

#[cfg(target_family = "windows")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PhysicalFileId(u32, u64);

#[cfg(target_family = "windows")]
#[inline]
fn get_physical_file_id(
    metadata: &dyn std::os::windows::fs::MetadataExt,
) -> Option<PhysicalFileId> {
    let volume_id = metadata.volume_serial_number()?;
    let file_index = metadata.file_index()?;
    Some(PhysicalFileId(volume_id, file_index))
}

/// Uniquely identifies a file on the host computer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FileId(u32);

impl FileId {
    /// Indicates no association to any file
    pub const NONE: Self = Self(0);

    /// Returns true if this ID is not associated with a file
    #[inline]
    pub fn is_none(self) -> bool {
        self == Self::NONE
    }
}

/// A file containing source code
#[derive(Debug)]
pub struct SourceFile {
    path: Box<Path>,
    text: SharedStr,
}

impl SourceFile {
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

    #[inline]
    pub(crate) fn text_clone(&self) -> SharedStr {
        self.text.clone()
    }

    /// The length of the text content
    #[inline]
    pub fn text_len(&self) -> u32 {
        self.text.len() as u32
    }
}

enum AddFileError {
    TooManyFilesRegistered,
    FileTooLarge,
}

/// An error that can occur when registering a file
#[derive(Debug)]
pub enum RegisterFileError {
    /// Registering another file would exceed the maximum allowed number of registered files
    TooManyFilesRegistered,
    /// The files being registered exceeds the maximum allowed file size
    FileTooLarge,
    /// The path being registered does not point to a file
    NotAFile,
    /// An IO error occurred while registering the file
    IoError(std::io::Error),
}

impl From<AddFileError> for RegisterFileError {
    #[inline]
    fn from(value: AddFileError) -> Self {
        match value {
            AddFileError::TooManyFilesRegistered => Self::TooManyFilesRegistered,
            AddFileError::FileTooLarge => Self::FileTooLarge,
        }
    }
}

/// An error that can occur when registering an in-memory file
#[derive(Debug)]
pub enum RegisterMemoryFileError {
    /// Registering another file would exceed the maximum allowed number of registered files
    TooManyFilesRegistered,
    /// The files being registered exceeds the maximum allowed file size
    FileTooLarge,
    /// An in-memory file with the same name has already been registered
    NameAlreadyRegistered(FileId),
}

impl From<AddFileError> for RegisterMemoryFileError {
    #[inline]
    fn from(value: AddFileError) -> Self {
        match value {
            AddFileError::TooManyFilesRegistered => Self::TooManyFilesRegistered,
            AddFileError::FileTooLarge => Self::FileTooLarge,
        }
    }
}

impl From<std::io::Error> for RegisterFileError {
    #[inline]
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value)
    }
}

type HashMap<K, V> = ahash::AHashMap<K, V>;

/// Platform independant wrapper for reading unique files
pub struct FileServer {
    files: Vec<SourceFile>,
    memory_map: HashMap<Box<Path>, FileId>,
    physical_map: HashMap<PhysicalFileId, FileId>,
}

impl FileServer {
    /// Creates a new file server.
    pub fn new() -> Self {
        let mut files = Vec::new();

        let none_path: &Path = "<none>".as_ref();
        files.push(SourceFile {
            path: none_path.to_owned().into_boxed_path(),
            text: "".into(),
        });

        Self {
            files,
            memory_map: HashMap::new(),
            physical_map: HashMap::new(),
        }
    }

    /// The files registered by the server
    #[inline]
    pub fn files(&self) -> impl Iterator<Item = &SourceFile> {
        self.files.iter()
    }

    /// Gets a file that has previously been registered with the server.
    #[inline]
    pub fn get_file(&self, id: FileId) -> Option<&SourceFile> {
        self.files.get(id.0 as usize)
    }

    fn add_file(&mut self, path: Box<Path>, text: SharedStr) -> Result<FileId, AddFileError> {
        let id =
            u32::try_from(self.files.len()).map_err(|_| AddFileError::TooManyFilesRegistered)?;

        if text.len() >= (u32::MAX as usize) {
            return Err(AddFileError::FileTooLarge);
        }

        self.files.push(SourceFile { path, text });
        Ok(FileId(id))
    }

    /// Registers a file with the server and returns its ID.
    /// If the file has previously been registered no actual file system access will occur.
    ///
    /// If an in-memory file with the given name has been registered previously, it will be returned over any physical file.
    pub fn register_file<P: AsRef<Path>>(&mut self, path: P) -> Result<FileId, RegisterFileError> {
        let path = path.as_ref();
        if let Some(&id) = self.memory_map.get(path) {
            return Ok(id);
        }

        #[cfg(any(target_family = "unix", target_family = "windows"))]
        {
            let metadata = std::fs::metadata(path)?;
            let phys_id = get_physical_file_id(&metadata).ok_or(RegisterFileError::NotAFile)?;

            if let Some(&id) = self.physical_map.get(&phys_id) {
                Ok(id)
            } else {
                let text = std::fs::read_to_string(&path)?;
                let id = self.add_file(path.into(), text.into())?;
                self.physical_map.insert(phys_id, id);
                Ok(id)
            }
        }

        #[cfg(not(any(target_family = "unix", target_family = "windows")))]
        Err(RegisterFileError::IoError(
            std::io::ErrorKind::Unsupported.into(),
        ))
    }

    /// Registers an in-memory file with the server and returns its ID.
    pub fn register_file_memory<P, S>(
        &mut self,
        name: P,
        text: S,
    ) -> Result<FileId, RegisterMemoryFileError>
    where
        P: AsRef<Path>,
        S: Into<Cow<'static, str>>,
    {
        let name = name.as_ref();
        if let Some(&id) = self.memory_map.get(name) {
            return Err(RegisterMemoryFileError::NameAlreadyRegistered(id));
        }

        let text: SharedStr = match text.into() {
            Cow::Borrowed(text) => text.into(),
            Cow::Owned(text) => text.into(),
        };

        let id = self.add_file(name.into(), text)?;
        self.memory_map.insert(name.into(), id);
        Ok(id)
    }
}
