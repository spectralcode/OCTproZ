/*
 * Copyright © 2018 Martino Pilia <martino.pilia@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mesh.h"


/*!
 * \brief Create a vertex array object (VAO) with given vertices and indices.
 * \param gl_context OpenGL context for the buffers.
 * \param vertices Vector of vertices, as strided x-y-z coordinates.
 * \param indices Indices for the mesh.
 */
Mesh::Mesh(const std::vector<GLfloat>& vertices, const std::vector<GLuint> &indices)
	: m_vertices_count {vertices.size()}
	, m_indices_count {indices.size()}
{
	initializeOpenGLFunctions();

	// Generates and populates a VBO for the vertices
	glDeleteBuffers(1, &(m_vertex_VBO));
	glGenBuffers(1, &(m_vertex_VBO));
	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

	// Generates and populates a VBO for the element indices
	glDeleteBuffers(1, &(m_index_VBO));
	glGenBuffers(1, &(m_index_VBO));
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_VBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);

	// Creates a vertex array object (VAO) for drawing the mesh
	glDeleteVertexArrays(1, &(m_vao));
	glGenVertexArrays(1, &(m_vao));
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertex_VBO);
	glEnableVertexAttribArray(POSITION);
	glVertexAttribPointer(POSITION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_VBO);
	glBindVertexArray(0);
}


/*!
 * \brief Destructor.
 */
Mesh::~Mesh()
{
}


/*!
 * \brief Render the mesh.
 */
void Mesh::paint(void)
{
	glBindVertexArray(m_vao);
	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m_indices_count), GL_UNSIGNED_INT, nullptr);
	glBindVertexArray(0);
}
