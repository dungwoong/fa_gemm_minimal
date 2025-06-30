// taken from common/util.cuh

namespace cuTK {

template <int default_alignment=16>
struct shared_allocator {
    int *ptr;

    private:
        
        // this will RECURSIVELY define a variable-length array
        // this is actually some crazy programming
        template<typename A, size_t... dims>
        struct variadic_array;
        template<typename A, size_t first_dim, size_t... rest_dims>
        struct variadic_array<A, first_dim, rest_dims...> {
            using type = typename variadic_array<A, rest_dims...>::type[first_dim];
        };
        template<typename A>
        struct variadic_array<A> {
            using type = A;
        };
        template<typename A, size_t... dims> 
        using variadic_array_t = typename variadic_array<A, dims...>::type;

        template<int alignment>
        __device__ inline void align_ptr() {
            if constexpr (alignment > 0) {
                uint64_t p = reinterpret_cast<uint64_t>(ptr);
                if(p % alignment != 0) {
                    ptr = (int*)(p + (alignment-(p%alignment)));
                }
            }
        }
    public:
        /**
         * Constructor, give pointer to the allocator
         */
        __device__ shared_allocator(int *_ptr): ptr(_ptr) {}

        template<typename A, size_t... dims> 
        __device__ inline variadic_array_t<A, dims...>& allocate() {
            // static_assert(sizeof(A) % default_alignment == 0, "Type is not aligned properly for array allocation");
            align_ptr<default_alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }
};

}