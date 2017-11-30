#pragma once
#include <sparse> //eigen sparse matrix

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <complex>
#include <stdexcept>
#include <string>

/// <summary>
/// Compressed sparse row format is the default storage type.
/// Interfacing with eigen to manipulate host side data.
/// </summary>
template<typename T>
class CudaSparseMatrix 
{
public:
	CudaSparseMatrix() = default;
	~CudaSparseMatrix() = default;
	
private: //members
	Eigen::SparseMatrix<T, Eigen::RowMajor, int> m_A;

	//Handles and library objects
	cublasHandle_t cublasH = NULL;
	cusparseHandle_t cusparseH = NULL;
	cudaStream_t stream = NULL;
	cusparseMatDescr_t descrA = NULL;

private: //error checking
	//Error types of the used libraries
	cublasStatus_t cublasStat = CUBLAS_STATUS_SUCCESS;
	cusparseStatus_t cusparseStat = CUSPARSE_STATUS_SUCCESS;
	cudaError_t cudaStat = cudaSuccess;

	void check_cublas_status(cublasStatus_t status) {
#ifdef
		if (status != CUBLAS_STATUS_SUCCESS)
			throw std::runtime_error("cublas error: " + std::to_string(status));
#endif
	}
	void check_cusparse_status(cusparseStatus_t status) {
#ifdef
		if (status != CUSPARSE_STATUS_SUCCESS)
			throw std::runtime_error("cusparse error: " + std::to_string(status));
#endif
	}
	void check_cuda_status(cudaError_t status) {
#ifdef
		if (status != cudaSuccess)
			throw std::runtime_error("cuda error: " + std::to_string(status));
#endif
	}

public:
	CudaSparseMatrix(const CudaSparseMatrix& copy_this) = delete;
	CudaSparseMatrix& operator=(const CudaSparseMatrix& copy_this) = delete;

	CudaSparseMatrix(CudaSparseMatrix&& move_this) = delete;
	CudaSparseMatrix& operator=(CudaSparseMatrix&& move_this) = delete;

	template<typename Q = T>
	using resolve_float = typename std::enable_if<std::is_same<Q, float>::value, bool>::type;

	template<typename Q = T>
	using resolve_double = typename std::enable_if<std::is_same<Q, double>::value, bool>::type;

	template<typename Q = T>
	using resolve_complex_float = typename std::enable_if<std::is_same<Q, std::complex<float>>::value, bool>::type;

	template<typename Q = T>
	using resolve_complex_double = typename std::enable_if<std::is_same<Q, std::complex<double>>::value, bool>::type;

	template<typename Q = T>
	using resolve_all = typename std::enable_if<
		!std::is_same<Q, float>::value &&
		!std::is_same<Q, double>::value &&
		!std::is_same<Q, std::complex<float>>::value, void>::value &&
		!std::is_same<Q, std::complex<double>>::value, void>::value;
};