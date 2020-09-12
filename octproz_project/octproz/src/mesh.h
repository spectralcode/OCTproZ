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

#pragma once

#include <QOpenGLExtraFunctions>

/*!
 * \brief Class to represent a mesh.
 */
class Mesh : protected QOpenGLExtraFunctions
{
public:
	Mesh(const std::vector<GLfloat>& vertices, const std::vector<GLuint>& indices);
	virtual ~Mesh();

	void paint(void);

private:
	GLuint m_vao {0};
	GLuint m_vertex_VBO {0};
	GLuint m_normal_VBO {0};
	GLuint m_index_VBO {0};
	size_t m_vertices_count {0};
	size_t m_indices_count {0};

	enum AttributeArray {POSITION=0, NORMAL=1, TEXCOORD=2};
};
