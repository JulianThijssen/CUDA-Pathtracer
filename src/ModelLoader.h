#pragma once
#ifndef MESHLOADER_H
#define MESHLOADER_H

#include "Scene.h"
#include "Mesh.h"
#include <string>
#include <cstdlib>
#include <fstream>
#include <vector>

std::vector<std::string> split(std::string s, char* delimiter) {
	std::vector<std::string> tokens;

	char* line_c = new char[s.size() + 1];
	std::copy(s.begin(), s.end(), line_c);
	line_c[s.size()] = '\0';

	char* ctok = strtok(line_c, delimiter);
	while (ctok != NULL) {
		tokens.push_back(std::string(ctok));
		ctok = strtok(NULL, delimiter);
	}
	delete[] line_c;

	return tokens;
}

void loadMesh(Scene &scene, std::string path) {
	std::ifstream f("res/cornell_box.obj");

	//std::vector<Mesh*> meshes;
	Mesh* m = new Mesh();
	std::vector<Vector3f> vertices;
	std::vector<Vector3f> normals;
	std::vector<Face> faces;

	for (std::string line; std::getline(f, line);) {
		if (line.size() < 1) {
			m->vertices = new Vector3f[vertices.size()];
			m->normals = new Vector3f[normals.size()];
			m->faces = new Face[faces.size()];
			for (int i = 0; i < vertices.size(); i++) {
				m->vertices[i].set(vertices[i].x, vertices[i].y, vertices[i].z);
			}
			for (int i = 0; i < normals.size(); i++) {
				m->normals[i].set(normals[i].x, normals[i].y, normals[i].z);
			}
			for (int i = 0; i < faces.size(); i++) {
				m->faces[i].v0 = faces[i].v0;
				m->faces[i].v1 = faces[i].v1;
				m->faces[i].v2 = faces[i].v2;
				m->faces[i].n0 = faces[i].n0;
				m->faces[i].n1 = faces[i].n1;
				m->faces[i].n2 = faces[i].n2;
				printf("faces: %d, %d, %d\n", m->faces[i].v0, m->faces[i].v1, m->faces[i].v2);
			}
			m->numVerts = (unsigned int) vertices.size();
			m->numNorms = (unsigned int) normals.size();
			m->numFaces = (unsigned int) faces.size();
			scene.addMesh(*m);
			printf("Presizes: %d, %d, %d, %f\n", m->numVerts, m->numNorms, m->numFaces, m->emission);
			m = new Mesh();
			vertices.clear();
			normals.clear();
			faces.clear();
			continue;
		}

		std::vector<std::string> tokens = split(line, " ");

		if ("o" == tokens[0]) {
			printf("Name: %s\n", tokens[1].c_str());
		}
		if ("c" == tokens[0]) {
			float x = (float) atof(tokens[1].c_str());
			float y = (float) atof(tokens[2].c_str());
			float z = (float) atof(tokens[3].c_str());
			m->albedo.set(x, y, z);
			m->emission = (float) atof(tokens[4].c_str());
		}
		if ("v" == tokens[0] || "vn" == tokens[0]) {
			float x = (float) atof(tokens[1].c_str());
			float y = (float) atof(tokens[2].c_str());
			float z = (float) atof(tokens[3].c_str());

			if ("v" == tokens[0]) {
				vertices.push_back(Vector3f(x, y, z));
			}
			if ("vn" == tokens[0]) {
				normals.push_back(Vector3f(x, y, z));
			}
		}
		if ("f" == tokens[0]) {
			Face face;
			
			std::vector<std::string> faceTokens;
			faceTokens = split(tokens[1], "/");
			face.v0 = atoi(faceTokens[0].c_str()) - 1;
			face.n0 = atoi(faceTokens[1].c_str()) - 1;

			faceTokens = split(tokens[2], "/");
			face.v1 = atoi(faceTokens[0].c_str()) - 1;
			face.n1 = atoi(faceTokens[1].c_str()) - 1;

			faceTokens = split(tokens[3], "/");
			face.v2 = atoi(faceTokens[0].c_str()) - 1;
			face.n2 = atoi(faceTokens[1].c_str()) - 1;

			faces.push_back(face);
		}
	}
	f.close();

	scene.meshCount = (unsigned int) scene.meshes.size();
}

#endif /* MESHLOADER_H */
