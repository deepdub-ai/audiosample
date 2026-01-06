"""
HTTPRangeFile: A file-like object that fetches data from HTTP/S URLs using Range requests.
Provides seekable random access to remote files without downloading the entire content.
"""

import io
from typing import Optional, Dict, Tuple
from logging import getLogger

logger = getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None  # type: ignore


class HTTPRangeFile(io.IOBase):
    """
    A file-like object that fetches data from HTTP/S URLs using Range requests.
    Implements seek/read/tell for random access to remote files.
    
    This class provides a seekable file interface over HTTP, using byte-range 
    requests to fetch only the data that's needed. It includes block-aligned 
    caching to minimize the number of HTTP requests when reading sequentially
    or seeking within cached regions.
    
    Parameters
    ----------
    url : str
        The HTTP or HTTPS URL to fetch data from.
    block_size : int, optional
        Size of cached blocks in bytes. Larger blocks mean fewer HTTP requests
        but more memory usage. Default is 256KB.
    session : requests.Session, optional
        A requests Session to use for HTTP requests. If not provided, a new
        session will be created. Using a shared session enables connection pooling.
    headers : dict, optional
        Additional headers to include in all HTTP requests (e.g., for authentication).
    timeout : float, optional
        Timeout for HTTP requests in seconds. Default is 30 seconds.
    max_cache_blocks : int, optional
        Maximum number of blocks to keep in cache. Oldest blocks are evicted
        when this limit is reached. Default is 64 (16MB with default block_size).
    
    Raises
    ------
    ImportError
        If the requests library is not installed.
    ValueError
        If the server does not support Range requests.
    requests.HTTPError
        If the HTTP request fails.
    
    Examples
    --------
    >>> f = HTTPRangeFile("https://example.com/audio.mp3")
    >>> f.seek(1000)
    1000
    >>> data = f.read(100)
    >>> len(data)
    100
    >>> f.close()
    """
    
    def __init__(
        self, 
        url: str, 
        block_size: int = 256 * 1024,
        session: Optional["requests.Session"] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        max_cache_blocks: int = 64
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "The 'requests' library is required for HTTP Range support. "
                "Install it with: pip install requests"
            )
        
        super().__init__()
        self.url = url
        self.block_size = block_size
        self._session = session
        self._owns_session = session is None
        self.timeout = timeout
        self.max_cache_blocks = max_cache_blocks
        
        # Build headers with sensible defaults
        self.base_headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; AudioSample/1.0; +https://github.com/deepdub-ai/audiosample)',
            'Accept': '*/*',
        }
        if headers:
            self.base_headers.update(headers)
        
        self._pos = 0
        self._size: Optional[int] = None
        self._block_cache: Dict[int, bytes] = {}
        self._cache_order: list = []  # LRU tracking
        self._closed = False
        self._supports_ranges = True
        self._full_content: Optional[bytes] = None  # Cache for servers that don't support ranges
        self._initialized = False  # Lazy initialization flag
    
    @property
    def session(self) -> "requests.Session":
        """Get or create the requests session."""
        if self._session is None:
            self._session = requests.Session()
        return self._session
    
    def _extract_size_from_response(self, resp: "requests.Response") -> bool:
        """
        Extract file size from response headers.
        Returns True if size was determined.
        """
        # Check if server supports range requests
        accept_ranges = resp.headers.get('Accept-Ranges', '').lower()
        if accept_ranges == 'none':
            self._supports_ranges = False
        
        # Try Content-Range header first (from 206 response): "bytes 0-99/12345"
        content_range = resp.headers.get('Content-Range', '')
        if '/' in content_range:
            try:
                size_str = content_range.split('/')[-1]
                if size_str != '*':
                    self._size = int(size_str)
                    return True
            except ValueError:
                pass
        
        # For 200 responses, Content-Length is the full file size
        if resp.status_code == 200:
            content_length = resp.headers.get('Content-Length')
            if content_length:
                self._size = int(content_length)
                return True
        
        return False
    
    def _fetch_range(self, start: int, end: int) -> bytes:
        """
        Fetch a byte range from the server.
        
        Parameters
        ----------
        start : int
            Start byte position (inclusive).
        end : int
            End byte position (inclusive).
        
        Returns
        -------
        bytes
            The fetched data.
        """
        # If we already have the full file cached (server doesn't support ranges), use it
        if self._full_content is not None:
            return self._full_content[start:end + 1]
        
        headers = {**self.base_headers, 'Range': f'bytes={start}-{end}'}
        
        resp = self.session.get(
            self.url, 
            headers=headers, 
            allow_redirects=True,
            timeout=self.timeout
        )
        
        # Extract size from response headers (lazy initialization)
        if self._size is None:
            self._extract_size_from_response(resp)
        
        if resp.status_code == 200:
            # Server returned full content instead of range
            # This means ranges aren't properly supported - cache the full content
            self._supports_ranges = False
            self._full_content = resp.content
            if self._size is None:
                self._size = len(self._full_content)
            return self._full_content[start:end + 1]
        elif resp.status_code == 206:
            self._initialized = True
            return resp.content
        else:
            resp.raise_for_status()
            return b''  # unreachable, but for type checker
    
    def _get_block(self, block_num: int) -> bytes:
        """
        Fetch a block, using cache if available.
        
        Uses LRU eviction when cache is full.
        """
        if block_num in self._block_cache:
            # Move to end of LRU list
            if block_num in self._cache_order:
                self._cache_order.remove(block_num)
            self._cache_order.append(block_num)
            return self._block_cache[block_num]
        
        start = block_num * self.block_size
        if self._size is not None:
            end = min(start + self.block_size - 1, self._size - 1)
        else:
            end = start + self.block_size - 1
        
        if self._size is not None and start >= self._size:
            return b''
        
        data = self._fetch_range(start, end)
        
        # Evict old blocks if cache is full
        while len(self._block_cache) >= self.max_cache_blocks and self._cache_order:
            old_block = self._cache_order.pop(0)
            self._block_cache.pop(old_block, None)
        
        self._block_cache[block_num] = data
        self._cache_order.append(block_num)
        
        return data
    
    def read(self, size: int = -1) -> bytes:
        """
        Read up to size bytes from the current position.
        
        Parameters
        ----------
        size : int, optional
            Number of bytes to read. If -1, read until end of file.
        
        Returns
        -------
        bytes
            The data read.
        """
        self._check_closed()
        
        if self._size is not None and self._pos >= self._size:
            return b''
        
        read_all = size < 0
        
        if read_all:
            if self._size is not None:
                size = self._size - self._pos
            else:
                # Unknown size - we'll read until EOF
                size = float('inf')
        
        if size == 0:
            return b''
        
        result = bytearray()
        bytes_remaining = size
        
        if self._size is not None and not read_all:
            bytes_remaining = min(bytes_remaining, self._size - self._pos)
        
        while bytes_remaining > 0:
            # After first block fetch, we should know the size
            if read_all and self._size is not None and bytes_remaining == float('inf'):
                bytes_remaining = self._size - self._pos
                if bytes_remaining <= 0:
                    break
            
            block_num = self._pos // self.block_size
            offset_in_block = self._pos % self.block_size
            
            block = self._get_block(block_num)
            if not block:
                break
            
            available = len(block) - offset_in_block
            if available <= 0:
                break
            
            to_read = min(bytes_remaining, available) if bytes_remaining != float('inf') else available
            result.extend(block[offset_in_block:offset_in_block + to_read])
            self._pos += to_read
            if bytes_remaining != float('inf'):
                bytes_remaining -= to_read
        
        return bytes(result)
    
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """
        Seek to a position in the file.
        
        Parameters
        ----------
        offset : int
            Position offset.
        whence : int, optional
            Reference point: SEEK_SET (0), SEEK_CUR (1), or SEEK_END (2).
        
        Returns
        -------
        int
            The new absolute position.
        """
        self._check_closed()
        
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            if self._size is None:
                raise io.UnsupportedOperation("Cannot seek from end without known size")
            self._pos = self._size + offset
        else:
            raise ValueError(f"Invalid whence value: {whence}")
        
        # Clamp position to valid range
        self._pos = max(0, self._pos)
        if self._size is not None:
            self._pos = min(self._pos, self._size)
        
        return self._pos
    
    def tell(self) -> int:
        """Return current position."""
        self._check_closed()
        return self._pos
    
    def seekable(self) -> bool:
        """Return True if the stream supports seeking."""
        # Seekable if server supports ranges OR if we have full content cached
        return self._supports_ranges or self._full_content is not None
    
    def readable(self) -> bool:
        """Return True if the stream can be read from."""
        return True
    
    def writable(self) -> bool:
        """Return False - HTTP streams are read-only."""
        return False
    
    def _check_closed(self) -> None:
        """Raise ValueError if the file is closed."""
        if self._closed:
            raise ValueError("I/O operation on closed file")
    
    def close(self) -> None:
        """Close the file and release resources."""
        if not self._closed:
            self._closed = True
            self.clear_cache()
            if self._owns_session and self._session is not None:
                self._session.close()
                self._session = None
    
    @property
    def closed(self) -> bool:
        """Return True if the file is closed."""
        return self._closed
    
    def __len__(self) -> int:
        """Return the file size in bytes."""
        return self._size or 0
    
    @property
    def size(self) -> Optional[int]:
        """Return the file size in bytes, or None if unknown."""
        return self._size
    
    def clear_cache(self) -> None:
        """Clear the internal block cache."""
        self._block_cache.clear()
        self._cache_order.clear()
        self._full_content = None
    
    def __enter__(self) -> "HTTPRangeFile":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        size_str = f"{self._size} bytes" if self._size else "unknown size"
        return f"<HTTPRangeFile [{status}] {self.url!r} ({size_str})>"

